#!/bin/bash
#
#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/StructTokenBench_utility.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/StructTokenBench_utility.%j.err # STDERR


export DIR=/n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/StructTokenBench
## esm
# tokenizer=WrappedESM3Tokenizer
# tokenizername=esm3
# d_model=128
# lr=0.001
# EXTRA_MODEL_ARGS=""

## using VanillaVQ
ckpt_name="VanillaVQ"
path="$DIR/struct_token_bench_release_ckpt/codebook_512x1024-1e+19-PST-last.ckpt/checkpoint/mp_rank_00_model_states.pt"
quantizer_use_linear_project=false

# general extra arguments besides $SHARED_ARGS
tokenizer=WrappedOurPretrainedTokenizer
tokenizername=ourpretrained_${ckpt_name}
d_model=1024
lr=0.001
quantizer_codebook_size=512
EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$path tokenizer_ckpt_name=${ckpt_name} quantizer_codebook_size=$quantizer_codebook_size quantizer_codebook_embed_size=$d_model model_encoder_dout=$d_model quantizer_use_linear_project=$quantizer_use_linear_project"

# CASP14
target_field=null task_goal="codebook_utilization" experiment_prefix="${task_goal}_casp14"

# CAMEO
target_field=null task_goal="codebook_utilization" experiment_prefix="${task_goal}_cameo"

EXTRA_TASK_ARGS="test_only=true model.task_goal=${task_goal} experiment_name=${experiment_prefix}_${tokenizername} optimization.micro_batch_size=8"

SHARED_ARGS="tokenizer=$tokenizer model.d_model=$d_model trainer.devices=[0] optimization.optimizer.lr=$lr data.target_field=$target_field experiment_name=${experiment_prefix}_${tokenizername}_lr${lr} run_name=tryout_test default_data_dir=$DIR/struct_token_bench_release_data/ data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ trainer.default_root_dir=$DIR/struct_token_bench_logs/ ${EXTRA_TASK_ARGS} ${EXTRA_MODEL_ARGS}"

# casp14
# CUDA_VISIBLE_DEVICES=0 python ./src/script/run_supervised_task.py --config-name=casp14.yaml  $SHARED_ARGS

# cameo
CUDA_VISIBLE_DEVICES=0 python ./src/script/run_supervised_task.py --config-name=cameo.yaml  $SHARED_ARGS