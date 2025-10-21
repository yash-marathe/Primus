#!/usr/bin/env python3

import argparse
import math
import os

from examples.scripts.utils import log_error_and_exit, log_info
from primus.core.launcher.parser import PrimusParser

# Memory constants
BYTES_PER_MB = 1024 * 1024
BYTES_PER_FP16 = 2
BYTES_PER_FP32 = 4
MEMORY_OVERHEAD = 1.05  # 5% overhead buffer

# ===== Parameter Memory (weights + optimizer states) =====

def _attn_term(args):
    """Return the attention parameter term for one transformer layer."""
    # Group-query & multi-latent attention support.
    # If GQA not enabled, fall back to per-head queries.
    num_query_groups = args.num_query_groups if args.group_query_attention and args.num_query_groups else args.num_attention_heads

    # Projection ratio: (kv_channels * n_heads) / hidden_size
    query_proj_to_hidden = (args.kv_channels * args.num_attention_heads) / args.hidden_size

    if args.multi_latent_attention:
        # q_term: either dense or LoRA factored Q with RoPE/Q-norm
        if args.q_lora_rank is None:
            q_term = args.hidden_size * args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
        else:
            q_term = args.q_lora_rank * (
                args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim) + 1
            )
        attn = (
            q_term
            # kv lora + rope + kv norm
            + args.kv_lora_rank * (
                args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1
            )
            # pos emb
            + args.hidden_size * args.qk_pos_emb_head_dim
            # out proj
            + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
        )
        return attn

    # Standard attention path (Q,K,V,O projections)
    return (
        2 * args.hidden_size * args.hidden_size *
        ((1 + (num_query_groups / args.num_attention_heads)) * query_proj_to_hidden)
    )


def _dense_layer_params(args, attn_term):
    """Dense transformer layer parameter count (includes 2 layer norms)."""
    gated_mult = 1.5 if args.swiglu else 1.0
    return 2 * args.hidden_size * (args.ffn_hidden_size * gated_mult + 2) + attn_term


def _moe_layer_params(args, attn_term):
    """MoE transformer layer parameter count (regular + shared experts + 2 layer norms)."""
    if args.num_experts is None:
        return _dense_layer_params(args, attn_term)
    gated_mult = 1.5 if args.swiglu else 1.0
    shared_sz = 0 if args.moe_shared_expert_intermediate_size is None else args.moe_shared_expert_intermediate_size
    moe_ffn = args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    return 2 * args.hidden_size * (
        (moe_ffn * args.num_experts * gated_mult) + (shared_sz * gated_mult) + 2
    ) + attn_term


def _layer_pattern_counts(args):
    """Return (#dense_layers, #moe_layers) for the main stack, and mtp tail counts if used."""
    if args.num_experts is None:
        moe_pattern = [0] * args.num_layers
    else:
        if isinstance(args.moe_layer_freq, int):
            moe_pattern = [1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)]
        else:
            moe_pattern = list(args.moe_layer_freq)
            assert len(moe_pattern) == args.num_layers, (
                f"Invalid moe_layer_freq length: {len(moe_pattern)} (expected {args.num_layers})"
            )
    num_moe = sum(moe_pattern)
    num_dense = args.num_layers - num_moe

    # MTP tail layers: if present, tail block matches type of last main layer
    mtp_dense = 0
    mtp_moe = 0
    if args.mtp_num_layers is not None:
        last_is_moe = (moe_pattern[-1] == 1) if len(moe_pattern) > 0 else 0
        mtp_moe = args.mtp_num_layers if last_is_moe else 0
        mtp_dense = args.mtp_num_layers if not last_is_moe else 0

    return num_dense, num_moe, mtp_dense, mtp_moe


def _embedding_params(args):
    return args.hidden_size * args.padded_vocab_size


def _final_ln_params(args):
    return 2 * args.hidden_size


def _most_loaded_shard_params(args, block_params, mtp_params, embedding_params):
    """PP×TP-aware most-loaded shard parameters."""
    params_in_block_per_shard = block_params / max(1, args.pipeline_model_parallel_size)
    params_on_most_loaded = (params_in_block_per_shard + mtp_params + embedding_params) / max(1, args.tensor_model_parallel_size)

    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        # The untied output head shard lives alongside embeddings when PP=1
        params_on_most_loaded += embedding_params / max(1, args.tensor_model_parallel_size)

    return params_on_most_loaded


def compute_parameter_and_optimizer_memory(args, verbose=False):
    """Compute bytes for parameters + optimizer on the most loaded shard."""
    attn = _attn_term(args)
    dense_layer = _dense_layer_params(args, attn)
    moe_layer = _moe_layer_params(args, attn)

    n_dense, n_moe, mtp_dense, mtp_moe = _layer_pattern_counts(args)

    block_params = dense_layer * n_dense + moe_layer * n_moe + _final_ln_params(args)
    mtp_params = dense_layer * mtp_dense + moe_layer * mtp_moe
    embedding_params = _embedding_params(args)

    most_loaded_params = _most_loaded_shard_params(args, block_params, mtp_params, embedding_params)

    if args.untie_embeddings_and_output_weights:
        print(f"Untied output head adds another embedding-sized param matrix.")
        embedding_params *= 2
    total_params = block_params + mtp_params + embedding_params

    if verbose:
        print(f"Dense layer params (billions): {dense_layer/1e9:.3f}")
        print(f"MoE layer params (billions):   {moe_layer/1e9:.3f}")
        print(f"Transformer block params (B):  {block_params/1e9:.3f}")
        if args.mtp_num_layers:
            print(f"MTP block params (B):         {mtp_params/1e9:.3f}")
        print(f"Embedding params (B):          {embedding_params/1e9:.3f}")
        print(f"Total params (B):              {total_params/1e9:.3f}")
        print(f"Most-loaded shard params (B):  {most_loaded_params/1e9:.3f}")

    # Optimizer state bytes per parameter:
    #  - ZeRO off: 18 bytes (fp16 param + master fp32 + Adam moments)
    #  - Distributed optimizer (ZeRO style): 6 + 12/DP bytes
    bytes_per_param = 18 if not args.use_distributed_optimizer else 6 + (12 / max(1, args.data_parallel_size))

    return most_loaded_params * bytes_per_param

# ===== Activation Memory =====

def activation_memory_with_sp(args, num_microbatches=None, verbose=False):
    """Calculate activation memory."""
    # Base per-layer activations
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / BYTES_PER_MB / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Inputs to embedding (pp_size microbatches in flight)
    activation_memory += 8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size

    # Dropout in embedding layer (pp_size microbatches in flight)
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Interleaved schedule penalty
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # Non-interleaved discount when microbatches < pp_size
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    # Output layer + CE loss when PP=1
    if args.pipeline_model_parallel_size == 1:
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    # TP partitioning
    activation_memory /= args.tensor_model_parallel_size

    return activation_memory

def activation_memory_without_sp(args, num_microbatches=None, verbose=False):
    """Precise non-SP path with proper pipeline scheduling and interleaving."""
    per_layer = args.seq_length * args.micro_batch_size * args.hidden_size * (
        10 + (24 / max(1, args.tensor_model_parallel_size))
    )
    if verbose:
        print(f"Activation memory per transformer layer (no SP): {per_layer / BYTES_PER_MB:.1f} MB")

    total = per_layer * args.num_layers

    # Embedding inputs (pp_size microbatches in flight) + embedding dropout
    total += 8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    total += args.seq_length * args.micro_batch_size * args.hidden_size * args.pipeline_model_parallel_size

    # Interleaved schedule penalty
    if args.virtual_pipeline_model_parallel_size is not None:
        penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        if verbose:
            print(f"Interleaved schedule memory penalty: {penalty:.2f}")
        total *= penalty

    # Non-interleaved schedule: discount if fewer microbatches than pp stages
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            total *= min(1.0, num_microbatches / args.pipeline_model_parallel_size)
            if verbose:
                print(f"In-flight microbatches: {min(num_microbatches, args.pipeline_model_parallel_size)}")
        else:
            if verbose:
                print(f"In-flight microbatches: {args.pipeline_model_parallel_size}")

    # If PP=1, include logits and final LN outputs (TP-sharded logits)
    if args.pipeline_model_parallel_size == 1:
        logits = args.seq_length * args.micro_batch_size * args.padded_vocab_size
        logits /= max(1, args.tensor_model_parallel_size)
        final_ln = args.seq_length * args.micro_batch_size * args.hidden_size
        total += (logits + final_ln) * 2  # bytes

    # 5% overhead
    total *= MEMORY_OVERHEAD

    return total

def parse_args():
    p = argparse.ArgumentParser(description="Analytical memory model for Primus pretrain config.")
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--verbose', action='store_true')
    # TODO: automatically extract from real tokenizer
    p.add_argument('--padded-vocab-size', type=int, default=52000)
    cli_args = p.parse_args()

    log_info("Parsed arguments for config: %s" % cli_args.config)
    world_size = int(os.environ.get('NNODES')) * int(os.environ.get('GPUS_PER_NODE'))
    log_info(f"World size: {world_size}")
    config_parser = PrimusParser()
    primus_config = config_parser.parse(cli_args)
    args = primus_config.get_module_config('pre_trainer')
    args.verbose = cli_args.verbose

    # Derived defaults
    if args.kv_channels is None:
        args.kv_channels = args.hidden_size // args.num_attention_heads
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size * 4
    if args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.hidden_size * 4
    if not args.group_query_attention:
        # If GQA not set, treat as per-head queries
        args.num_query_groups = args.num_attention_heads

    args.padded_vocab_size = cli_args.padded_vocab_size
    if not hasattr(args, 'data_parallel_size') or args.data_parallel_size is None:
        args.data_parallel_size = world_size // (
            args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
        )
    if args.num_layers_per_virtual_pipeline_stage is None and args.num_virtual_stages_per_pipeline_rank is None:
        args.virtual_pipeline_model_parallel_size = None
    elif args.num_layers_per_virtual_pipeline_stage is not None:
        args.virtual_pipeline_model_parallel_size = args.num_layers // (
            args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size
        )
    else:
        args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
    return args


def main():
    args = parse_args()

    # ----- Parameters + optimizer -----
    param_opt_bytes = compute_parameter_and_optimizer_memory(args, args.verbose)

    # ----- Activations -----
    if args.sequence_parallel and args.recompute_granularity == 'selective':
        act_bytes = activation_memory_with_sp(args, args.micro_batch_size, args.verbose)
    else:
        act_bytes = activation_memory_without_sp(args, args.micro_batch_size, args.verbose)

    # ----- Report -----
    param_mb = param_opt_bytes / BYTES_PER_MB
    act_mb = act_bytes / BYTES_PER_MB
    total_mb = param_mb + act_mb

    print("\nAnalytical Memory Requirements:")
    print(f"Parameters and optimizer: {param_mb:.2f} MB")
    print(f"Activations: {act_mb:.2f} MB")
    print(f"Total: {total_mb:.2f} MB ({total_mb / 1024:.2f} GB)")


if __name__ == '__main__':
    main()