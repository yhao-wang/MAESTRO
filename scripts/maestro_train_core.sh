#!/usr/bin/env bash
# MAESTRO. Defaults reproduce the paper configuration: K=5,
# two teacher takeovers, and three paragraphs per takeover.
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAESTRO_RUNTIME_DIR="${MAESTRO_RUNTIME_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL is required}
TEACHER_MODEL=${TEACHER_MODEL:?TEACHER_MODEL is required}
TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA is required}
BENCH=${BENCH:?BENCH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
EXP_ID=${EXP_ID:-$(basename "${OUTPUT_DIR}")}

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/training.log") 2>&1
cd "${MAESTRO_RUNTIME_DIR}"

export MATH_GRADER_PATH=${MATH_GRADER_PATH:-${MAESTRO_RUNTIME_DIR}/maestro/reward/grader}
export PYTHONPATH=${MAESTRO_RUNTIME_DIR}/maestro/patches/vllm:${PYTHONPATH:-}

export MAESTRO_RUNTIME_VLLM_PATCH=1
export MAESTRO_RUNTIME_VLLM_PATCH_DEFER=1
# Log-probability stability knobs (see MAESTRO_LOGPROB_STABILITY fix, 2026-09-16):
#   * compute per-token log-probs from fp32 logits instead of the bf16 fallback
#   * floor non-finite log-probs instead of letting them become inf loss
#   * keep k1 distillation losses clamped
export MAESTRO_RUNTIME_LOGPROB_FP32=${MAESTRO_RUNTIME_LOGPROB_FP32:-1}
export MAESTRO_RUNTIME_LOGPROB_FP32_CHUNK=${MAESTRO_RUNTIME_LOGPROB_FP32_CHUNK:-4096}
export MAESTRO_RUNTIME_LOGPROB_FLOOR=${MAESTRO_RUNTIME_LOGPROB_FLOOR:--30.0}
export MAESTRO_RUNTIME_LOGPROB_DIAG=${MAESTRO_RUNTIME_LOGPROB_DIAG:-1}
export MAESTRO_LOSS_MAX_CLAMP=${MAESTRO_LOSS_MAX_CLAMP:-10.0}
export MAESTRO_LOGPROB_MIN_CLAMP=${MAESTRO_LOGPROB_MIN_CLAMP:--10.0}
export MAESTRO_RUNTIME_ROLLOUT_MODE=${MAESTRO_RUNTIME_ROLLOUT_MODE:-relay}
export MAESTRO_RUNTIME_DRAFT_SAMPLING=1
export MAESTRO_RUNTIME_COLLECT_DRAFT_PROBS=1
export MAESTRO_RUNTIME_SOURCE_METRICS=${MAESTRO_RUNTIME_SOURCE_METRICS:-1}
export MAESTRO_RUNTIME_TOKENIZER_PATH=${STUDENT_MODEL}
export MAESTRO_TRIGGER_TOPK=${MAESTRO_TRIGGER_TOPK:-5}
export MAESTRO_MAX_TAKEOVERS=${MAESTRO_MAX_TAKEOVERS:-2}
export MAESTRO_PARAGRAPHS_PER_TAKEOVER=${MAESTRO_PARAGRAPHS_PER_TAKEOVER:-3}
export MAESTRO_PDS_AGGREGATION=${MAESTRO_PDS_AGGREGATION:-rolling}
export MAESTRO_PDS_PREFIX_ALPHA=${MAESTRO_PDS_PREFIX_ALPHA:-0.5}
export MAESTRO_PDS_PREFIX_TAU=${MAESTRO_PDS_PREFIX_TAU:-12}
export MAESTRO_PDS_PARAGRAPH_MIN_TOKENS=${MAESTRO_PDS_PARAGRAPH_MIN_TOKENS:-16}
export MAESTRO_PDS_GRAY_UPPER=${MAESTRO_PDS_GRAY_UPPER:-0.06}
export MAESTRO_PDS_PARAGRAPH_THRESHOLD=${MAESTRO_PDS_PARAGRAPH_THRESHOLD:-0.12}
export MAESTRO_PDS_PARAGRAPH_MIN_GAP=${MAESTRO_PDS_PARAGRAPH_MIN_GAP:-0.05}
export MAESTRO_PDS_PARAGRAPH_LAST16_THRESHOLD=${MAESTRO_PDS_PARAGRAPH_LAST16_THRESHOLD:-0.09}
export MAESTRO_PDS_FALLBACK_PATIENCE=${MAESTRO_PDS_FALLBACK_PATIENCE:-2}
export MAESTRO_MAX_TAKEOVER_TOKENS=${MAESTRO_MAX_TAKEOVER_TOKENS:-256}
export MAESTRO_PDS_ENABLE=${MAESTRO_PDS_ENABLE:-0}
export MAESTRO_PDS_TRACE=${MAESTRO_PDS_TRACE:-${MAESTRO_PDS_ENABLE}}
export MAESTRO_PDS_TOPK=${MAESTRO_PDS_TOPK:-16}
export MAESTRO_PDS_THRESHOLD=${MAESTRO_PDS_THRESHOLD:-0.45}
export MAESTRO_PDS_WINDOW_TOKENS=${MAESTRO_PDS_WINDOW_TOKENS:-100}
export MAESTRO_PDS_WINDOW_STRIDE=${MAESTRO_PDS_WINDOW_STRIDE:-${MAESTRO_PDS_WINDOW_TOKENS}}
export MAESTRO_PDS_PATIENCE=${MAESTRO_PDS_PATIENCE:-2}
export MAESTRO_PDS_WARMUP_TOKENS=${MAESTRO_PDS_WARMUP_TOKENS:-256}
export MAESTRO_MAX_TAKEOVERS_BEHAVIOR=${MAESTRO_MAX_TAKEOVERS_BEHAVIOR:-stop}
if [[ "${MAESTRO_PDS_ENABLE}" =~ ^(1|true|yes|y|on)$ ]]; then
    export MAESTRO_TRIGGER_MODE=${MAESTRO_TRIGGER_MODE:-pds_only}
    export MAESTRO_RUNTIME_TRACE_TOPK=${MAESTRO_RUNTIME_TRACE_TOPK:-${MAESTRO_PDS_TOPK}}
else
    export MAESTRO_TRIGGER_MODE=${MAESTRO_TRIGGER_MODE:-reflection_only}
fi

LOSS_MODE=${LOSS_MODE:-maestro}
MAESTRO_ACTION_TOKEN=emitted

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export MAESTRO_RUNTIME_PLATFORM_PORT_ISOLATION=${MAESTRO_RUNTIME_PLATFORM_PORT_ISOLATION:-1}
export MAESTRO_RUNTIME_MASTER_PORT_BASE=${MAESTRO_RUNTIME_MASTER_PORT_BASE:-45000}
export MAESTRO_RUNTIME_MASTER_PORT_WIDTH=${MAESTRO_RUNTIME_MASTER_PORT_WIDTH:-64}
export MAESTRO_RUNTIME_VLLM_PORT_BASE=${MAESTRO_RUNTIME_VLLM_PORT_BASE:-48000}
export MAESTRO_RUNTIME_TRACE_SUMMARY_EVERY=${MAESTRO_RUNTIME_TRACE_SUMMARY_EVERY:-10}
export MAESTRO_RUNTIME_SPECULATIVE_ENABLE=1
export MAESTRO_RUNTIME_TARGET_MODEL=${TEACHER_MODEL}
export MAESTRO_RUNTIME_DRAFT_MODEL=${STUDENT_MODEL}
export MAESTRO_RUNTIME_NUM_SPECULATIVE_TOKENS=${MAESTRO_RUNTIME_NUM_SPECULATIVE_TOKENS:-4}
export MAESTRO_RUNTIME_DRAFT_TENSOR_PARALLEL_SIZE=${MAESTRO_RUNTIME_DRAFT_TENSOR_PARALLEL_SIZE:-1}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
actor_gpus_per_node=${ACTOR_GPUS_PER_NODE:-4}
teacher_gpus_per_node=${TEACHER_GPUS_PER_NODE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-1}
actor_sp=${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
val_max_response_length=${VAL_MAX_RESPONSE_LENGTH:-32768}
val_max_model_len=$((max_prompt_length + val_max_response_length + 1))
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-${val_max_model_len}}
teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-${val_max_model_len}}

rollout_tp=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}
teacher_tp=${TEACHER_TENSOR_MODEL_PARALLEL_SIZE:-1}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
teacher_gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION:-0.45}
actor_ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}
teacher_max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-4096}
actor_checkpoint_save_contents=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:--1}
val_before_train=${VAL_BEFORE_TRAIN:-False}
total_epochs=${TOTAL_EPOCHS:-1}
maestro_trace_dir=${MAESTRO_TRACE_DIR:-${OUTPUT_DIR}/maestro_trace}
python_bin=${PYTHON_BIN:-python3}

echo "[MAESTRO] experiment=${EXP_ID} output=${OUTPUT_DIR}"
echo "[MAESTRO] student=${STUDENT_MODEL} teacher=${TEACHER_MODEL}"
echo "[MAESTRO] mode=${MAESTRO_RUNTIME_ROLLOUT_MODE} trigger_topk=${MAESTRO_TRIGGER_TOPK} max_takeovers=${MAESTRO_MAX_TAKEOVERS} paragraphs=${MAESTRO_PARAGRAPHS_PER_TAKEOVER}"
echo "[MAESTRO] trigger_mode=${MAESTRO_TRIGGER_MODE} pds_enable=${MAESTRO_PDS_ENABLE} pds_topk=${MAESTRO_PDS_TOPK} pds_tau=${MAESTRO_PDS_THRESHOLD} pds_window=${MAESTRO_PDS_WINDOW_TOKENS} pds_stride=${MAESTRO_PDS_WINDOW_STRIDE} pds_patience=${MAESTRO_PDS_PATIENCE} pds_warmup=${MAESTRO_PDS_WARMUP_TOKENS}"
echo "[MAESTRO] loss=${LOSS_MODE} action_token=${MAESTRO_ACTION_TOKEN}"
echo "[MAESTRO] response=${max_response_length} rollout_tp=${rollout_tp} teacher_tp=${teacher_tp} draft_tp=${MAESTRO_RUNTIME_DRAFT_TENSOR_PARALLEL_SIZE}"
echo "[MAESTRO] resources=actor:${actor_gpus_per_node}+teacher:${teacher_gpus_per_node} GPUs"
echo "[MAESTRO] python=${python_bin}"
echo "[MAESTRO] platform_port_isolation=${MAESTRO_RUNTIME_PLATFORM_PORT_ISOLATION} master_base=${MAESTRO_RUNTIME_MASTER_PORT_BASE} vllm_base=${MAESTRO_RUNTIME_VLLM_PORT_BASE}"

printenv | sort > "${OUTPUT_DIR}/run_config.env"
"${python_bin}" - "${OUTPUT_DIR}/run_config.md" <<'PY'
import os
import sys

path = sys.argv[1]
keys = [
    "EXP_ID", "OUTPUT_DIR", "STUDENT_MODEL", "TEACHER_MODEL", "TRAIN_DATA", "BENCH",
    "MAESTRO_RUNTIME_ROLLOUT_MODE", "MAESTRO_TRIGGER_MODE", "MAESTRO_TRIGGER_TOPK",
    "MAESTRO_PDS_ENABLE", "MAESTRO_PDS_TRACE", "MAESTRO_PDS_TOPK",
    "MAESTRO_PDS_THRESHOLD", "MAESTRO_PDS_WINDOW_TOKENS", "MAESTRO_PDS_WINDOW_STRIDE",
    "MAESTRO_PDS_PATIENCE", "MAESTRO_PDS_WARMUP_TOKENS",
    "MAESTRO_PDS_AGGREGATION", "MAESTRO_PDS_PREFIX_ALPHA", "MAESTRO_PDS_PREFIX_TAU",
    "MAESTRO_PDS_PARAGRAPH_MIN_TOKENS", "MAESTRO_PDS_GRAY_UPPER",
    "MAESTRO_PDS_PARAGRAPH_THRESHOLD", "MAESTRO_PDS_PARAGRAPH_MIN_GAP",
    "MAESTRO_PDS_PARAGRAPH_LAST16_THRESHOLD", "MAESTRO_PDS_FALLBACK_PATIENCE",
    "MAESTRO_MAX_TAKEOVERS_BEHAVIOR",
    "MAESTRO_MAX_TAKEOVERS", "MAESTRO_PARAGRAPHS_PER_TAKEOVER",
    "MAESTRO_MAX_TAKEOVER_TOKENS", "MAESTRO_RUNTIME_NUM_SPECULATIVE_TOKENS",
    "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "VAL_MAX_RESPONSE_LENGTH",
    "ROLLOUT_MAX_MODEL_LEN", "TEACHER_MAX_MODEL_LEN", "TRAIN_BATCH_SIZE",
    "PPO_MINI_BATCH_SIZE", "ACTOR_GPUS_PER_NODE", "TEACHER_GPUS_PER_NODE",
    "ROLLOUT_GPU_MEMORY_UTILIZATION", "TEACHER_GPU_MEMORY_UTILIZATION",
    "TEACHER_MAX_NUM_BATCHED_TOKENS", "LOSS_MODE", "MAESTRO_ACTION_TOKEN",
    "SAVE_FREQ", "TEST_FREQ", "VAL_BEFORE_TRAIN", "TOTAL_EPOCHS",
    "MAESTRO_STOP_AFTER_STEP",
    "MAESTRO_RUNTIME_TRACE_SUMMARY_EVERY", "MAESTRO_RUNTIME_TRACE_TOPK", "PYTHON_BIN",
    "MAESTRO_RUNTIME_LOGPROB_FP32", "MAESTRO_RUNTIME_LOGPROB_FP32_CHUNK", "MAESTRO_RUNTIME_LOGPROB_FLOOR",
    "MAESTRO_RUNTIME_LOGPROB_DIAG", "MAESTRO_LOSS_MAX_CLAMP", "MAESTRO_LOGPROB_MIN_CLAMP",
]
with open(path, "w", encoding="utf-8") as f:
    f.write("# MAESTRO run config\n\n")
    for key in keys:
        f.write(f"- {key}={os.environ.get(key, '')}\n")
PY

"${python_bin}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${TRAIN_DATA}']" \
    "data.val_files=['${BENCH}/aime-24_verl.parquet','${BENCH}/aime-2025_verl.parquet']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.seed=42 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE:-constant} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO:-0.0} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${LR_MIN_RATIO:-0.1} \
    actor_rollout_ref.actor.optim.num_cycles=${LR_NUM_CYCLES:-0.5} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.gradient_accumulation_steps=${grad_accum_steps} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${actor_sp} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.n=32 \
    ++actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    ++actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    ++actor_rollout_ref.rollout.val_kwargs.max_tokens=${val_max_response_length} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len_per_gpu} \
    trainer.logger=console \
    trainer.project_name=maestro \
    trainer.experiment_name=${EXP_ID} \
    trainer.n_gpus_per_node=${actor_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    +trainer.maestro_trace_dir="${maestro_trace_dir}" \
    ++trainer.validation_data_dir="${OUTPUT_DIR}/val_generations" \
    +actor_rollout_ref.actor.checkpoint.save_contents="${actor_checkpoint_save_contents}" \
    reward.reward_manager.source=register \
    reward.reward_manager.name=remote \
    reward.num_workers=4 \
    reward.custom_reward_function.path=${MAESTRO_RUNTIME_DIR}/maestro/reward/math_reward.py \
    reward.custom_reward_function.name=compute_score \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${teacher_gpus_per_node} \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_memory_utilization} \
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len} \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens} \
    distillation.distillation_loss.loss_mode=${LOSS_MODE} \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.loss_max_clamp=${MAESTRO_LOSS_MAX_CLAMP:-10.0} \
    distillation.distillation_loss.log_prob_min_clamp=${MAESTRO_LOGPROB_MIN_CLAMP:--10.0} \
    +distillation.distillation_loss.maestro_action_token=${MAESTRO_ACTION_TOKEN} \
    "$@"

echo "=== MAESTRO training complete: ${EXP_ID} ==="
