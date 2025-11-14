export DIR='./'
tokenizer_list=(WrappedESM3Tokenizer WrappedFoldSeekTokenizer WrappedProTokensTokenizer WrappedProteinMPNNTokenizer WrappedMIFTokenizer WrappedCheapS1D64Tokenizer WrappedAIDOTokenizer)
tokenizer_name_list=(esm3 foldseek protokens proteinmpnn mif cheapS1D64 aido)
dmodel_list=(128 2 32 128 256 64 384)

# for i in "${!tokenizer_list[@]}"
# do
i=2
tokenizer=${tokenizer_list[i]}
tokenizername=${tokenizer_name_list[i]}
d_model=${dmodel_list[i]}
echo $tokenizer, $d_model
for lr in "0.1" "0.01" "0.001" "0.0001" "0.00005" "0.00001" "0.000005" "0.000001";
do
    echo $lr, "bindint_${tokenizername}_lr${lr}"
    EXTRA_MODEL_ARGS=""
    
    target_field="binding_label" experiment_prefix="bindint"
    EXTRA_TASK_ARGS=""        
    
    SHARED_ARGS="tokenizer=$tokenizer model.d_model=$d_model trainer.devices=[0] optimization.optimizer.lr=$lr data.target_field=$target_field experiment_name=${experiment_prefix}_${tokenizername}_lr${lr} run_name=tryout_test default_data_dir=$DIR/struct_token_bench_release_data/ data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ trainer.default_root_dir=$DIR/struct_token_bench_logs/ ${EXTRA_TASK_ARGS} ${EXTRA_MODEL_ARGS}"

    CUDA_VISIBLE_DEVICES=0 python ./src/script/run_supervised_task.py --config-name=interpro.yaml $SHARED_ARGS
    
    # task-specific python command
done
# done