#!/bin/bash
#
#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 600g # memory pool for all cores
#SBATCH --gres=gpu:4 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/StructTokenBench_pretrain.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/StructTokenBench_pretrain.%j.err # STDERR

# for debug, run command: sbatch -p gpu_test --gres=gpu:1 -t 0-12:00 pretrain.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 {time to sleep} {codebook size} {fastdev}"
  exit 1
fi

module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01

export DIR=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/StructTokenBench
CKPT_DIR=$DIR/struct_token_bench_release_ckpt

# vanillavq
# use_linear_project=false
# freeze_codebook=false
# model_name="VanillaVQ"
sleep $1 # to avoid version conflicts
echo "sleep finish"
# aminoaseed
use_linear_project=true
freeze_codebook=true
model_name="AminoAseed"

warmup_step=5426
total_step=108530
lr=0.0001

if [ $3 -eq 1 ]; then
  fast_dev=true
else
  fast_dev=false
fi

validate_only=false
freeze_codebook=false
_need_init=true
# pretrained_ckpt_path="/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/StructTokenBench/struct_token_bench_release_ckpt/codebook_512x1024-1e+19-linear-fixed-last.ckpt/checkpoint/mp_rank_00_model_states.pt" # ''
pretrained_ckpt_path=''

export NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DEVICES=$(seq -s, 0 $((NGPU-1)))
PYTHONPATH=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/StructTokenBench \
CUDA_LAUNCH_BLOCKING=1 \
HYDRA_FULL_ERROR=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
/n/holylfs06/LABS/mzitnik_lab/Users/msun415/envs/pstbench/bin/python -m src.script.run_pretraining_vqvae --config-name=pretrain.yaml tokenizer=WrappedESM3Tokenizer trainer.devices=$NGPU optimization.micro_batch_size=4 optimization.scheduler.num_warmup_steps=${warmup_step} max_steps=${total_step} optimization.optimizer.lr=$lr optimization.scheduler.plateau_ratio=0.0 lightning.callbacks.checkpoint.monitor="validation_bb_rmsd" lightning.callbacks.checkpoint.mode="min" lightning.callbacks.checkpoint.save_top_k=1 trainer.log_every_n_steps=512 data.fast_dev_run=${fast_dev} data.data_version=mmcif_files_filtered_subsample10 experiment_name=vqvae-pretrain-subsample10_${model_name}_fastdev${fast_dev} run_name=test model.quantizer.use_linear_project=${use_linear_project} model.quantizer.freeze_codebook=${freeze_codebook} model.quantizer.codebook_size=${2} model.quantizer._need_init=${_need_init} model.pretrained_ckpt_path=${pretrained_ckpt_path} model.ckpt_path='' validate_only=${validate_only} default_data_dir=$DIR/struct_token_bench_release_data/ data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ trainer.default_root_dir=$DIR/struct_token_bench_logs/