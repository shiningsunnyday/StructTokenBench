#!/bin/bash
#
#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH -c 16 # number of cores
#SBATCH --mem 100g # memory pool for all cores
#SBATCH --gres=gpu:4 # gpu
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/StructTokenBench_pretrain.%j.out # STDOUT
#SBATCH -e /n/holylfs06/LABS/mzitnik_lab/Users/msun415/foldingdiff/scripts/slurm/StructTokenBench_pretrain.%j.err # STDERR

# for debug, run command: sbatch -p gpu_test --gres=gpu:1 -t 0-12:00 pretrain.sh

module load cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01

export DIR='./'
CKPT_DIR=$DIR/struct_token_bench_release_ckpt

## esm
tokenizer=WrappedESM3Tokenizer
tokenizername=esm3
d_model=128
lr=0.001
EXTRA_MODEL_ARGS=""

## vanilla vq
# ckpt_name="VanillaVQ"
# path="$DIR/struct_token_bench_release_ckpt/codebook_512x1024-1e+19-PST-last.ckpt/checkpoint/mp_rank_00_model_states.pt"
# quantizer_use_linear_project=false
# tokenizer=WrappedOurPretrainedTokenizer
# tokenizername=ourpretrained_${ckpt_name}
# d_model=1024
# lr=0.001
# quantizer_codebook_size=512
# EXTRA_MODEL_ARGS="tokenizer_pretrained_ckpt_path=$path tokenizer_ckpt_name=${ckpt_name} quantizer_codebook_size=$quantizer_codebook_size quantizer_codebook_embed_size=$d_model model_encoder_dout=$d_model quantizer_use_linear_project=$quantizer_use_linear_project"

## effectiveness
task="$1"

###############################
# 1. Task-specific variables  #
###############################
if   [[ $task == "bindint" ]];        then target_field="binding_label"     experiment_prefix="bindint"     EXTRA_TASK_ARGS=""
elif [[ $task == "bindbio" ]];        then target_field="binding_label"     experiment_prefix="bindbio"     EXTRA_TASK_ARGS=""
elif [[ $task == "catbio" ]];         then target_field="catalytic_label"   experiment_prefix="catbio"      EXTRA_TASK_ARGS=""
elif [[ $task == "catint" ]];         then target_field="activesite_label"  experiment_prefix="catint"      EXTRA_TASK_ARGS=""
elif [[ $task == "conserved" ]];      then target_field="conservedsite_label" experiment_prefix="con"       EXTRA_TASK_ARGS=""
elif [[ $task == "repeat" ]];         then target_field="repeat_label"      experiment_prefix="rep"         EXTRA_TASK_ARGS=""
elif [[ $task == "bindshake" ]];      then target_field="binding_site"      experiment_prefix="bindshake"   EXTRA_TASK_ARGS=""
elif [[ $task == "epitope" ]];        then target_field="epitope_label"     experiment_prefix="ept"         EXTRA_TASK_ARGS=""
elif [[ $task == "homo" ]];           then target_field="fold_label"        experiment_prefix="homo"        EXTRA_TASK_ARGS="optimization.micro_batch_size=64"
elif [[ $task == "FlexRMSF" ]];       then target_field="rmsf_score"        experiment_prefix="flexrmsf"    EXTRA_TASK_ARGS="data.pdb_data_dir=$DIR/struct_token_bench_release_data/data/physicochemical/ lightning.callbacks.checkpoint.monitor='validation_spearmanr'"
elif [[ $task == "FlexBFactor" ]];    then target_field="bfactor_score"     experiment_prefix="flexbfactor" EXTRA_TASK_ARGS="data.pdb_data_dir=$DIR/struct_token_bench_release_data/data/physicochemical/ lightning.callbacks.checkpoint.monitor='validation_spearmanr'"
elif [[ $task == "FlexNEQ" ]];        then target_field="neq_score"         experiment_prefix="flexneq"     EXTRA_TASK_ARGS="data.pdb_data_dir=$DIR/struct_token_bench_release_data/data/physicochemical/ lightning.callbacks.checkpoint.monitor='validation_spearmanr'"
else
  echo "Unknown task: $task"; exit 1
fi

################################
# 2. Arguments shared by all   #
################################
SHARED_ARGS="tokenizer=$tokenizer model.d_model=$d_model trainer.devices=[0] \
optimization.optimizer.lr=$lr data.target_field=$target_field \
experiment_name=${experiment_prefix}_${tokenizername}_lr${lr} run_name=tryout_test \
default_data_dir=$DIR/struct_token_bench_release_data/ \
data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ \
trainer.default_root_dir=$DIR/struct_token_bench_logs/ \
${EXTRA_TASK_ARGS} ${EXTRA_MODEL_ARGS}"

################################
# 3. Pick config & run script  #
################################
if   [[ $task == "bindshake" ]];                                then CONFIG="proteinshake_binding_site.yaml"
elif [[ $task == "repeat"   || $task == "bindint"  || \
       $task == "catint"   || $task == "conserved" ]];          then CONFIG="interpro.yaml"
elif [[ $task == "bindbio" || $task == "catbio" ]];             then CONFIG="biolip2.yaml"
elif [[ $task == "epitope" ]];                                  then CONFIG="proteinglue_epitope_region.yaml"
elif [[ $task == "homo" ]];                                     then CONFIG="remote_homology.yaml"
elif [[ $task == "FlexRMSF" || $task == "FlexBFactor" || \
       $task == "FlexNEQ" ]];                                   then CONFIG="atlas.yaml"
else
  echo "No config found for task: $task"; exit 1
fi

CUDA_VISIBLE_DEVICES=0 python ./src/script/run_supervised_task.py --config-name=$CONFIG $SHARED_ARGS