# export NCCL_IB_HCA=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export NCCL_IB_HCA=^mlx5_1,mlx5_6
export CPUS_PER_TASK=96
export HSA_NO_SCRATCH_RECLAIM=1 # change to 0
export NVTE_CK_USES_BWD_V3=1 # change to 0

export EXP="examples/megatron/configs/deepseek_1T-pretrain.yaml"

  # useocpm2m-401-038
  # useocpm2m-401-040
  # useocpm2m-401-045
  # useocpm2m-401-046

  # useocpm2m-401-084
  # useocpm2m-401-086

  # BAD NODES
  # useocpm2m-401-075

  # GOOD NODES
  # useocpm2m-401-067
  # useocpm2m-401-068

# GOOD NODES
ALL_NODES=(
  useocpm2m-401-025
  useocpm2m-401-028
  useocpm2m-401-054
  useocpm2m-401-056
  useocpm2m-401-073
  useocpm2m-401-075
  useocpm2m-401-108
  useocpm2m-401-118
  useocpm2m-401-121
  useocpm2m-401-123
  useocpm2m-401-124
  useocpm2m-401-125
  useocpm2m-401-137
  useocpm2m-401-139
  useocpm2m-401-140
  useocpm2m-401-142
  useocpm2m-401-144
  useocpm2m-401-147
)

# ALL_NODES=(
#   useocpm2m-401-139
#   useocpm2m-401-140
#   useocpm2m-401-142
#   useocpm2m-401-144
#   useocpm2m-401-146
#   useocpm2m-401-147
# )

# slurm number of nodes
ALL_NNODES=18

# the real number of nodes to run
export NNODES=16

SELECTED_NODES=("${ALL_NODES[@]:0:$ALL_NNODES}")
export NODELIST=$(IFS=, ; echo "${SELECTED_NODES[*]}")

MBS=1
TP=1
ETP=1

GBS=512
PP=4
EP=32
VPP=4
TOPK=8
OPTIMIZER=adam
RECOMPUTE_LAYERS=0
RECOMPUTE_ID_START=0
BALANCE=True


CONFIG="deepep.Turbo-attn-gg-deepep.mockdata.gc.BF16.GBS$GBS.PP$PP.EP$EP.CP$CP.VPP$VPP.TOPK$TOPK.rc-$RECOMPUTE_LAYERS.rcids-$RECOMPUTE_ID_START.nodes$NNODES.$OPTIMIZER.BALANCE-$BALANCE"
echo "config: $CONFIG"


if [ $VPP -gt 1 ]; then
    export VPP_CONFIG="--num_virtual_stages_per_pipeline_rank $VPP"
fi

export PRIMUS_WORKSPACE=output/poolside/1T
export PRIMUS_USER=yuankai
export PRIMUS_GROUP="date-$(date +%Y%m%d)"
export PRIMUS_EXP_NAME=$CONFIG
mkdir -p $PRIMUS_WORKSPACE

export GPU_MAX_HW_QUEUES=8


LOG_DIR=./$PRIMUS_WORKSPACE/$PRIMUS_GROUP/$PRIMUS_USER/$CONFIG/
export PRIMUS_PPDUMP_FILE=$LOG_DIR/pp_dump/
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/training.log
echo $LOG_FILE

EXPORT_CONFIG=$LOG_DIR/config.yaml
bash ./examples/run_slurm_pretrain.sh --micro_batch_size $MBS \
                                      --global_batch_size $GBS \
                                      --tensor_model_parallel_size $TP \
                                      --expert_tensor_parallel_size $ETP \
                                      --pipeline_model_parallel_size $PP \
                                      --expert_model_parallel_size $EP \
                                      --moe_router_force_load_balancing $BALANCE \
                                      --moe_router_topk $TOPK \
                                      --manual_gc True \
                                      --manual_gc_interval 1 \
                                      --optimizer $OPTIMIZER \
                                      --cp_comm_type a2a \
                                      --moe_use_legacy_grouped_gemm True \
                                      --use_turbo_token_dispatcher_fp8_alltoall null \
                                      --dump_pp_data False \
                                      --moe_router_num_groups 1 \
                                      --moe_router_group_topk 1 \
                                      ${VPP_CONFIG} \
                                      --pp_warmup True \
                                      --profile True \
                                      --record_shapes True \
                                      --disable_profiler_activity_cpu False \
                                      --use_pytorch_profiler True \
                                      --profile_step_start 5 \
                                      --profile_step_end 6 \
                                      --export-config $EXPORT_CONFIG \
                                      --num_layers 16 \
                                      --num_experts 448 \
                                      --moe_router_topk 16 \
                                      --train_iters 10 2>&1 | tee $LOG_FILE

                                    #   --recompute_layer_ids_start $RECOMPUTE_ID_START \
                                    #   --moe_permute_fusion True \
                                    #   --use_turbo_token_dispatcher_fp8_alltoall False \
