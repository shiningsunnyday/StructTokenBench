#!/bin/bash
#
#SBATCH -c 16 # number of cores
#SBATCH --mem 600g # memory pool for all cores
#SBATCH --gres=gpu:4 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o ../scripts/slurm/StructTokenBench_pretrain.%j.out # STDOUT
#SBATCH -e ../scripts/slurm/StructTokenBench_pretrain.%j.err # STDERR

# for debug, run command: sbatch -p gpu_test --gres=gpu:1 -t 0-12:00 pretrain.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 {time to sleep} {codebook size} {fastdev} {aminoaseed|vanillavq}"
  exit 1
fi

export DIR='./'
CKPT_DIR=$DIR/struct_token_bench_release_ckpt

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29527
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

CONDA_ENV=${CONDA_ENV:-pstbench}                 # name OR absolute path
CONDA_BASE=$(conda info --base)
PYTHON_BIN=${CONDA_PREFIX:-${CONDA_BASE}/envs/${CONDA_ENV}}/bin/python
export PYTHONPATH="$PWD:$PYTHONPATH"
runner="$PYTHON_BIN"

sleep $1 # to avoid version conflicts
echo "sleep finish"

if [ "$4" = "aminoaseed" ]; then
  # Current behavior
  use_linear_project=true
  freeze_codebook=true
  model_name="AminoAseed"
elif [ "$4" = "vanillavq" ]; then
  # Alternative behavior commented in your block
  use_linear_project=false
  freeze_codebook=false
  model_name="VanillaVQ"
else
  echo "Error: Invalid model parameter '$4'. Expected 'aminoaseed' or 'vanillavq'." >&2
  exit 1
fi

warmup_step=5426
total_step=108530
lr=0.0001

if [ $3 -eq 1 ]; then
  fast_dev=true
else
  fast_dev=false
fi

validate_only=false
_need_init=true
# pretrained_ckpt_path="./struct_token_bench_release_ckpt/codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/mp_rank_00_model_states.pt" # ''
pretrained_ckpt_path=''

export NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DEVICES=$(seq -s, 0 $((NGPU-1)))
PYTHONPATH=$DIR \
CUDA_LAUNCH_BLOCKING=1 \
HYDRA_FULL_ERROR=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
$runner -m src.script.run_pretraining_vqvae --config-name=pretrain.yaml tokenizer=WrappedESM3Tokenizer trainer.devices=$NGPU optimization.micro_batch_size=4 optimization.scheduler.num_warmup_steps=${warmup_step} max_steps=${total_step} optimization.optimizer.lr=$lr optimization.scheduler.plateau_ratio=0.0 lightning.callbacks.checkpoint.monitor="validation_bb_rmsd" lightning.callbacks.checkpoint.mode="min" lightning.callbacks.checkpoint.save_top_k=1 trainer.log_every_n_steps=512 data.fast_dev_run=${fast_dev} data.data_version=mmcif_files_filtered_subsample10 experiment_name=vqvae-pretrain-subsample10_${model_name}_fastdev${fast_dev} run_name=test model.quantizer.use_linear_project=${use_linear_project} model.quantizer.freeze_codebook=${freeze_codebook} model.quantizer.codebook_size=${2} model.quantizer._need_init=${_need_init} model.pretrained_ckpt_path=${pretrained_ckpt_path} model.ckpt_path='' validate_only=${validate_only} default_data_dir=$DIR/struct_token_bench_release_data/ data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ trainer.default_root_dir=$DIR/struct_token_bench_logs/