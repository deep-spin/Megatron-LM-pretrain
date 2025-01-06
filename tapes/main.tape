import "scslurm.tape"

global {
    ducttape_experimental_imports=true
    ducttape_experimental_submitters=true
    ducttape_experimental_multiproc=true
}

task PrepareModel
    > initial_model
    :: repo=@
    :: clip_original_dir=@
    :: model_name=@
    :: model_type=@
    :: tp=@
    :: pp=@
    :: .submitter=@
    :: .C=$prepare_C
    :: .account=$prepare_account
    :: .time=$prepare_time
    :: .cpus=$prepare_cpus
    :: .partition=$prepare_partition
    :: .gres=$prepare_gres
{
    # Download & convert CLIP model
    echo "Downloading & converting CLIP model..."
    python $repo/examples/multimodal/model_converter/clip_converter.py \
        --download-root $clip_original_dir \
        --output clip_mcore_dir \
        --tensor-parallel-size ${tp} \
        --use-te

    # Download & convert language model
    echo "Downloading & converting language model..."
    python $repo/examples/multimodal/download_hf_model.py \
        --model ${model_name} \
        --output-dir mistral_original_dir
    python $repo/tools/checkpoint/convert.py --model-type GPT \
        --loader llama_mistral \
        --saver mcore \
        --checkpoint-type hf \
        --model-size ${model_type} \
        --load-dir mistral_original_dir \
        --save-dir mistral_mcore_dir \
        --tokenizer-model ${model_name} \
        --target-tensor-parallel-size ${tp} \
        --target-pipeline-parallel-size ${pp} \
        --bf16

    # Combine models
    echo "Combining language and vision models..."
    bash $repo/examples/multimodal/combine_lm_vision_checkpoints.sh \
        mistral_mcore_dir \
        clip_mcore_dir \
        $initial_model

    # remove original and intermediate converted models to save space
    rm -rf mistral_original_dir
    rm -rf clip_mcore_dir mistral_mcore_dir
}

task PretrainModel
    < initial_model=@PrepareModel
    > model_dir
    :: repo=@
    :: tokenizer_model=$model_name
    :: model_type=@
    :: prompt_format=@
    :: image_preproc=@
    :: pixel_shuffle=@
    :: pretrain_dataset=@
    :: train_iters=$pretrain_iters
    :: batch_size=$pretrain_bsz
    :: lr=$pretrain_lr
    :: lr_warmup_fraction=$pretrain_lr_warmup
    :: unfreeze_lm=$pretrain_unfreeze_lm
    :: unfreeze_vit=$pretrain_unfreeze_vit
    :: save_interval=$pretrain_save_interval
    :: eval_interval=$pretrain_eval_interval
    :: external_resume=true
    :: external_model_dir=@
    :: external_tensorboard=@
    :: num_workers=@
    :: .submitter=@
    :: .C=$pretrain_C
    :: .account=$pretrain_account
    :: .time=$pretrain_time
    :: .cpus=$pretrain_cpus
    :: .gres=$pretrain_gres
{
    export NCCL_IB_SL=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    # if `save_external` is set, symlink it to the `model_dir`
    # and copy the config file to the `model_dir`
    if [ "$external_model_dir" != "" ]; then
        if [ "$external_resume" == false ]; then
            rm -rf $external_model_dir
        fi
        mkdir -p $external_model_dir
        ln -sf $external_model_dir $model_dir
    fi

    if [ "$external_tensorboard" != "" ]; then
        mkdir -p $external_tensorboard
        tensorboard=$external_tensorboard
    else
        mkdir -p tensorboard
        tensorboard=tensorboard
    fi

    export NVTE_APPLY_QK_LAYER_SCALING=0
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

    # define custom arguments based on model type
    if [ "$model_type" == "mistral-7B" ]; then
        # mistral-7B is the only model that requires the --disable-vision-class-token flag
        MODEL_ARGS="--language-model-type=mistral_7b"
        MODEL_ARGS="$MODEL_ARGS --disable-vision-class-token"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 8"
        MODEL_ARGS="$MODEL_ARGS --no-masked-softmax-fusion"
        MODEL_ARGS="$MODEL_ARGS --use-flash-attn"
        MODEL_ARGS="$MODEL_ARGS --untie-embeddings-and-output-weights"
        MODEL_ARGS="$MODEL_ARGS --disable-bias-linear"
        MODEL_ARGS="$MODEL_ARGS --position-embedding-type rope"
        MODEL_ARGS="$MODEL_ARGS --rotary-percent 1.0"
        MODEL_ARGS="$MODEL_ARGS --rotary-base 1000000"
        MODEL_ARGS="$MODEL_ARGS --swiglu"
        MODEL_ARGS="$MODEL_ARGS --attention-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --hidden-dropout 0.1"
        MODEL_ARGS="$MODEL_ARGS --tensor-model-parallel-size 4"
        MODEL_ARGS="$MODEL_ARGS --pipeline-model-parallel-size 1"
        MODEL_ARGS="$MODEL_ARGS --num-layers 32"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 4096"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 32"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 1024"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 4096"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 14336"
    elif [ "$model_type" == "qwen2.5-7B" ]; then
        MODEL_ARGS="--language-model-type=qwen2.5_7b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 4"
        MODEL_ARGS="$MODEL_ARGS --no-masked-softmax-fusion"
        MODEL_ARGS="$MODEL_ARGS --use-flash-attn"
        MODEL_ARGS="$MODEL_ARGS --untie-embeddings-and-output-weights"
        MODEL_ARGS="$MODEL_ARGS --disable-bias-linear"
        MODEL_ARGS="$MODEL_ARGS --add-qkv-bias"
        MODEL_ARGS="$MODEL_ARGS --position-embedding-type rope"
        MODEL_ARGS="$MODEL_ARGS --rotary-base 1000000"
        MODEL_ARGS="$MODEL_ARGS --swiglu"
        MODEL_ARGS="$MODEL_ARGS --no-bias-swiglu-fusion"
        MODEL_ARGS="$MODEL_ARGS --no-rope-fusion"
        MODEL_ARGS="$MODEL_ARGS --attention-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --hidden-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --tensor-model-parallel-size 4"
        MODEL_ARGS="$MODEL_ARGS --pipeline-model-parallel-size 1"
        MODEL_ARGS="$MODEL_ARGS --num-layers 28"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 3584"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 28"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 1024"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 131072"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 18944"
        MODEL_ARGS="$MODEL_ARGS --norm-epsilon 1e-6"
    elif [ "$model_type" == "eurollm-9B" ]; then
        # TODO: check if rotary-base and norm-epsilon are correct
        MODEL_ARGS="--language-model-type=eurollm_9b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 8"
        MODEL_ARGS="$MODEL_ARGS --no-masked-softmax-fusion"
        MODEL_ARGS="$MODEL_ARGS --use-flash-attn"
        MODEL_ARGS="$MODEL_ARGS --untie-embeddings-and-output-weights"
        MODEL_ARGS="$MODEL_ARGS --disable-bias-linear"
        MODEL_ARGS="$MODEL_ARGS --position-embedding-type rope"
        MODEL_ARGS="$MODEL_ARGS --rotary-base 10000"
        MODEL_ARGS="$MODEL_ARGS --swiglu"
        MODEL_ARGS="$MODEL_ARGS --attention-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --hidden-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --tensor-model-parallel-size 4"
        MODEL_ARGS="$MODEL_ARGS --pipeline-model-parallel-size 1"
        MODEL_ARGS="$MODEL_ARGS --num-layers 42"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 4096"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 32"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 1024"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 4096"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 12288"
    fi
    
    if [ "$image_preproc" == "basic" ]; then
        MODEL_ARGS="$MODEL_ARGS --seq-length 576"
    elif [ "$image_preproc" == "nvlm" ]; then
        MODEL_ARGS="$MODEL_ARGS --seq-length 256"  # Image embeddings sequence length
        MODEL_ARGS="$MODEL_ARGS --image-tag-type nvlm"
    else
        echo "Invalid image preproc: $image_preproc"
        exit 1
    fi

    if [ "$pixel_shuffle" == true ]; then
        MODEL_ARGS="$MODEL_ARGS --pixel-shuffle"
    fi

    torchrun --nproc_per_node 4 $repo/examples/multimodal/train.py \
        --apply-layernorm-1p \
        --attention-softmax-in-fp32 \
        --use-checkpoint-args \
        --use-distributed-optimizer \
        --transformer-impl transformer_engine \
        --use-te \
        $MODEL_ARGS \
        --train-iters ${train_iters} \
        --micro-batch-size 16 \
        --global-batch-size ${batch_size} \
        --lr-decay-iters ${train_iters} \
        --lr-warmup-fraction ${lr_warmup_fraction} \
        --lr ${lr} \
        --min-lr 1.0e-5 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 10 \
        --eval-interval ${eval_interval} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model ${tokenizer_model} \
        --tokenizer-prompt-format ${prompt_format} \
        --data-path $pretrain_dataset \
        --prompt-path ${repo}/examples/multimodal/manual_prompts.json \
        --save-interval ${save_interval} \
        --save ${model_dir} \
        --load ${model_dir} \
        --dataloader-save ${model_dir}/dataloader \
        --pretrained-checkpoint ${initial_model} \
        --num-workers $num_workers \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 1e-2 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.014 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
        --eod-mask-loss \
        $([ "$unfreeze_lm" == false ] && echo "--freeze-LM" || echo "") \
        $([ "$unfreeze_vit" == false ] && echo "--freeze-ViT" || echo "") \
        --patch-dim 14 \
        --img-h 336 \
        --img-w 336 \
        --dataloader-type external \
        --tensorboard-dir ${tensorboard} \
        --disable-vision-class-token \
        --distributed-timeout-minutes 60 \
        --allow-missing-vision-projection-checkpoint \
        --ckpt-format torch
}

func GenerateTestTask
    < model_dir
    > outputs
    :: repo
    :: tokenizer_model
    :: model_type
    :: prompt_format
    :: image_preproc
    :: pixel_shuffle
    :: input_image_path
    :: gt_path
    :: eval_bsz
    :: task
{
    export NCCL_IB_SL=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_APPLY_QK_LAYER_SCALING=0
    export PYTHONPATH="$PYTHONPATH:$repo/examples/multimodal" 
    
    # define custom arguments based on model type
    if [ "$model_type" == "mistral-7B" ]; then
        MODEL_ARGS="--language-model-type=mistral_7b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 8"
        MODEL_ARGS="$MODEL_ARGS --num-layers 32"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 4096"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 14336"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 32"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 4096"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 2048"
    elif [ "$model_type" == "qwen2.5-7B" ]; then
        MODEL_ARGS="--language-model-type=qwen2.5_7b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 4"
        MODEL_ARGS="$MODEL_ARGS --add-qkv-bias"
        MODEL_ARGS="$MODEL_ARGS --no-bias-swiglu-fusion"
        MODEL_ARGS="$MODEL_ARGS --no-rope-fusion"
        MODEL_ARGS="$MODEL_ARGS --num-layers 28"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 3584"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 28"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 131072"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 18944"
        MODEL_ARGS="$MODEL_ARGS --norm-epsilon 1e-6"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 2048"
    elif [ "$model_type" == "eurollm-9B" ]; then
        # TODO: check if rotary-base and norm-epsilon are correct
        MODEL_ARGS="--language-model-type=eurollm_9b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 8"
        MODEL_ARGS="$MODEL_ARGS --num-layers 42"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 4096"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 32"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 4096"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 12288"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 2048"
    fi

    if [ "$image_preproc" == "basic" ]; then
        MODEL_ARGS="$MODEL_ARGS --seq-length 576"
    elif [ "$image_preproc" == "nvlm" ]; then
        MODEL_ARGS="$MODEL_ARGS --seq-length 261"  # 256 image embeddings + 5 tile tag embeddings
        MODEL_ARGS="$MODEL_ARGS --use-tiling"
        MODEL_ARGS="$MODEL_ARGS --max-num-tiles 6"
        MODEL_ARGS="$MODEL_ARGS --use-thumbnail"
        MODEL_ARGS="$MODEL_ARGS --use-tile-tags"
        MODEL_ARGS="$MODEL_ARGS --image-tag-type nvlm"
    else
        echo "Invalid image preproc: $image_preproc"
        exit 1
    fi

    if [ "$pixel_shuffle" == true ]; then
        MODEL_ARGS="$MODEL_ARGS --pixel-shuffle"
    fi

    mkdir -p $outputs
    
    torchrun --nproc_per_node 4 $repo/examples/multimodal/run_text_generation.py \
        --apply-layernorm-1p \
        --attention-softmax-in-fp32 \
        --use-flash-attn \
        --transformer-impl transformer_engine \
        --use-te \
        --use-checkpoint-args \
        $MODEL_ARGS \
        --no-masked-softmax-fusion \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --rotary-base 1000000 \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 1 \
        --load ${model_dir} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model ${tokenizer_model} \
        --tokenizer-prompt-format ${prompt_format} \
        --bf16 \
        --micro-batch-size ${eval_bsz} \
        --out-seq-length 12 \
        --temperature 1.0 \
        --img-h 336 \
        --img-w 336 \
        --patch-dim 14 \
        --disable-vision-class-token \
        --seed 153 \
        --top_k 1 \
        --no-load-rng \
        --no-load-optim \
        --input-image-path ${input_image_path} \
        --num-partitions 0 \
        --partition-id 0 \
        --output-path $outputs/${task}_outputs \
        --gt-path $gt_path \
        --task ${task} \
        --num-frames 1 \
        --ckpt-format torch
}


task GenerateTestCoco calls GenerateTestTask
    < model_dir=@PretrainModel
    > outputs
    :: repo=@
    :: tokenizer_model=$model_name
    :: model_type=@
    :: prompt_format=@
    :: image_preproc=@
    :: pixel_shuffle=@
    :: eval_bsz=@
    :: task=captioning
    :: input_image_path=$coco_dir
    :: gt_path=$coco_gt_path
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres

task EvaluatePretrainedModel
    < outputs=$outputs@GenerateTestCoco
    > coco_results
    :: repo=@
    :: eval_bsz=@
    :: tokenizer_model=$model_name
    :: prompt_format=@
    :: coco_dir=@
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres
{
    # Evaluate results
    python $repo/examples/multimodal/evaluate_coco.py \
        --input-path $outputs/captioning_outputs \
        --groundtruth-path $coco_dir/coco_karpathy_test_gt.json \
        | tee $coco_results
}

task FineTuneModel
    < pretrained_dir=$model_dir@PretrainModel
    > finetuned_dir
    :: repo=@
    :: tokenizer_model=$model_name
    :: model_type=@
    :: prompt_format=@
    :: image_preproc=@
    :: pixel_shuffle=@
    :: sft_dataset=@
    :: train_iters=$finetune_iters
    :: batch_size=$finetune_bsz
    :: micro_batch_size=$finetune_micro_bsz
    :: lr=$finetune_lr
    :: lr_warmup_fraction=$finetune_lr_warmup
    :: unfreeze_vit=$finetune_unfreeze_vit
    :: save_interval=$finetune_save_interval
    :: eval_interval=$finetune_eval_interval
    :: external_resume=true
    :: external_model_dir=$finetune_model_dir
    :: external_tensorboard=$finetune_tensorboard
    :: nnodes=$finetune_nnodes
    :: gpus=$finetune_gpus
    :: num_workers=@
    :: master_addr=@
    :: master_port=@
    :: .submitter=@
    :: .C=$finetune_C
    :: .account=$finetune_account
    :: .time=$finetune_time
    :: .cpus=$finetune_cpus
    :: .nodes=$finetune_nodes
    :: .gres=$finetune_gres
    :: .qos=$finetune_qos
{
    export NCCL_IB_SL=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    # Handle external directories similar to PretrainModel
    if [ "$external_model_dir" != "" ]; then
        if [ "$external_resume" == false ]; then
            rm -rf $external_model_dir
        fi
        mkdir -p $external_model_dir
        ln -sf $external_model_dir $finetuned_dir
    fi

    if [ "$external_tensorboard" != "" ]; then
        mkdir -p $external_tensorboard
        tensorboard=$external_tensorboard
    else
        mkdir -p tensorboard
        tensorboard=tensorboard
    fi

    export NVTE_APPLY_QK_LAYER_SCALING=0
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

        # define custom arguments based on model type
    if [ "$model_type" == "mistral-7B" ]; then
        # mistral-7B is the only model that requires the --disable-vision-class-token flag
        MODEL_ARGS="--language-model-type=mistral_7b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 8"
        MODEL_ARGS="$MODEL_ARGS --no-masked-softmax-fusion"
        MODEL_ARGS="$MODEL_ARGS --use-flash-attn"
        MODEL_ARGS="$MODEL_ARGS --untie-embeddings-and-output-weights"
        MODEL_ARGS="$MODEL_ARGS --disable-bias-linear"
        MODEL_ARGS="$MODEL_ARGS --position-embedding-type rope"
        MODEL_ARGS="$MODEL_ARGS --rotary-percent 1.0"
        MODEL_ARGS="$MODEL_ARGS --rotary-base 1000000"
        MODEL_ARGS="$MODEL_ARGS --swiglu"
        MODEL_ARGS="$MODEL_ARGS --attention-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --hidden-dropout 0.1"
        MODEL_ARGS="$MODEL_ARGS --tensor-model-parallel-size 4"
        MODEL_ARGS="$MODEL_ARGS --pipeline-model-parallel-size 1"
        MODEL_ARGS="$MODEL_ARGS --num-layers 32"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 4096"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 32"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 2048"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 4096"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 14336"
    elif [ "$model_type" == "qwen2.5-7B" ]; then
        MODEL_ARGS="--language-model-type=qwen2.5_7b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 4"
        MODEL_ARGS="$MODEL_ARGS --no-masked-softmax-fusion"
        MODEL_ARGS="$MODEL_ARGS --use-flash-attn"
        MODEL_ARGS="$MODEL_ARGS --untie-embeddings-and-output-weights"
        MODEL_ARGS="$MODEL_ARGS --disable-bias-linear"
        MODEL_ARGS="$MODEL_ARGS --add-qkv-bias"
        MODEL_ARGS="$MODEL_ARGS --position-embedding-type rope"
        MODEL_ARGS="$MODEL_ARGS --rotary-base 1000000"
        MODEL_ARGS="$MODEL_ARGS --swiglu"
        MODEL_ARGS="$MODEL_ARGS --no-bias-swiglu-fusion"
        MODEL_ARGS="$MODEL_ARGS --no-rope-fusion"
        MODEL_ARGS="$MODEL_ARGS --attention-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --hidden-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --tensor-model-parallel-size 4"
        MODEL_ARGS="$MODEL_ARGS --pipeline-model-parallel-size 1"
        MODEL_ARGS="$MODEL_ARGS --num-layers 28"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 3584"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 28"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 2048"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 131072"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 18944"
        MODEL_ARGS="$MODEL_ARGS --norm-epsilon 1e-6"
    elif [ "$model_type" == "eurollm-9B" ]; then
        # TODO: check if rotary-base and norm-epsilon are correct
        MODEL_ARGS="--language-model-type=eurollm_9b"
        MODEL_ARGS="$MODEL_ARGS --normalization RMSNorm"
        MODEL_ARGS="$MODEL_ARGS --group-query-attention"
        MODEL_ARGS="$MODEL_ARGS --num-query-groups 8"
        MODEL_ARGS="$MODEL_ARGS --no-masked-softmax-fusion"
        MODEL_ARGS="$MODEL_ARGS --use-flash-attn"
        MODEL_ARGS="$MODEL_ARGS --untie-embeddings-and-output-weights"
        MODEL_ARGS="$MODEL_ARGS --disable-bias-linear"
        MODEL_ARGS="$MODEL_ARGS --position-embedding-type rope"
        MODEL_ARGS="$MODEL_ARGS --rotary-base 10000"
        MODEL_ARGS="$MODEL_ARGS --swiglu"
        MODEL_ARGS="$MODEL_ARGS --attention-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --hidden-dropout 0.0"
        MODEL_ARGS="$MODEL_ARGS --tensor-model-parallel-size 4"
        MODEL_ARGS="$MODEL_ARGS --pipeline-model-parallel-size 1"
        MODEL_ARGS="$MODEL_ARGS --num-layers 42"
        MODEL_ARGS="$MODEL_ARGS --hidden-size 4096"
        MODEL_ARGS="$MODEL_ARGS --num-attention-heads 32"
        MODEL_ARGS="$MODEL_ARGS --decoder-seq-length 2048"
        MODEL_ARGS="$MODEL_ARGS --max-position-embeddings 4096"
        MODEL_ARGS="$MODEL_ARGS --ffn-hidden-size 12288"
    fi

    if [ "$image_preproc" == "basic" ]; then
        MODEL_ARGS="$MODEL_ARGS --seq-length 576"
    elif [ "$image_preproc" == "nvlm" ]; then
        # NVLM-specific settings based on Qwen 72B SFT script
        MODEL_ARGS="$MODEL_ARGS --seq-length 261"  # 256 image embeddings + 5 tile tag embeddings
        MODEL_ARGS="$MODEL_ARGS --image-tag-type nvlm"
        # Tiling-specific arguments
        MODEL_ARGS="$MODEL_ARGS --use-tiling"
        MODEL_ARGS="$MODEL_ARGS --max-num-tiles 6"
        MODEL_ARGS="$MODEL_ARGS --use-thumbnail"
        MODEL_ARGS="$MODEL_ARGS --use-tile-tags"
    else
        echo "Invalid image preproc: $image_preproc"
        exit 1
    fi

    if [ "$pixel_shuffle" == true ]; then
        MODEL_ARGS="$MODEL_ARGS --pixel-shuffle"
    fi

    #export NCCL_ASYNC_ERROR_HANDLING=1
    distributed_args="--nnodes=$nnodes --nproc_per_node=$gpus"
    #distributed_args="${distributed_args} --rdzv_backend c10d --rdzv_endpoint $master_addr:$master_port"
    #distributed_args="${distributed_args} --node_rank=$SLURM_PROCID"

    torchrun $distributed_args $repo/examples/multimodal/train.py \
        --apply-layernorm-1p \
        --attention-softmax-in-fp32 \
        --use-checkpoint-args \
        --use-distributed-optimizer \
        --transformer-impl transformer_engine \
        --use-te \
        $MODEL_ARGS \
        --train-iters ${train_iters} \
        --micro-batch-size ${micro_batch_size} \
        --global-batch-size ${batch_size} \
        --lr-decay-iters ${train_iters} \
        --lr-warmup-fraction ${lr_warmup_fraction} \
        --lr ${lr} \
        --min-lr 1.0e-7 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 10 \
        --eval-interval ${eval_interval} \
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model ${tokenizer_model} \
        --tokenizer-prompt-format ${prompt_format} \
        --data-path $sft_dataset \
        --prompt-path ${repo}/examples/multimodal/manual_prompts.json \
        --save-interval ${save_interval} \
        --save ${finetuned_dir} \
        --load ${finetuned_dir} \
        --dataloader-save ${finetuned_dir}/dataloader \
        --pretrained-checkpoint ${pretrained_dir} \
        --num-workers $num_workers \
        --split 100,0,0 \
        --clip-grad 0.5 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.014 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
        --eod-mask-loss \
        $([ "$unfreeze_vit" == false ] && echo "--freeze-ViT" || echo "") \
        --patch-dim 14 \
        --img-h 336 \
        --img-w 336 \
        --dataloader-type external \
        --tensorboard-dir ${tensorboard} \
        --disable-vision-class-token \
        --distributed-timeout-minutes 60 \
        --ckpt-format torch
}

task GenerateTestMMMU calls GenerateTestTask
    < model_dir=$finetuned_dir@FineTuneModel
    > outputs
    :: repo=@
    :: tokenizer_model=$model_name
    :: model_type=@
    :: prompt_format=@
    :: image_preproc=@
    :: pixel_shuffle=@
    :: eval_bsz=@
    :: task=MMMU
    :: input_image_path=none
    :: gt_path=none
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres

task GenerateTestTextVQA calls GenerateTestTask
    < model_dir=$finetuned_dir@FineTuneModel
    > outputs
    :: repo=@
    :: tokenizer_model=$model_name
    :: model_type=@
    :: prompt_format=@
    :: image_preproc=@
    :: pixel_shuffle=@
    :: eval_bsz=@
    :: task=TextVQA
    :: input_image_path=$textvqa_dir
    :: gt_path=$textvqa_gt_path
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres

task GenerateTestAI2D calls GenerateTestTask
    < model_dir=$finetuned_dir@FineTuneModel
    > outputs
    :: repo=@
    :: tokenizer_model=$model_name
    :: model_type=@
    :: prompt_format=@
    :: image_preproc=@
    :: pixel_shuffle=@
    :: eval_bsz=@
    :: task=AI2D
    :: input_image_path=$ai2d_dir
    :: gt_path=$ai2d_gt_path
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres

task EvaluateFinetunedModel
    < ai2d_outputs=$outputs@GenerateTestAI2D
    < textvqa_outputs=$outputs@GenerateTestTextVQA
    < mmmu_outputs=$outputs@GenerateTestMMMU
    > ai2d_results
    > textvqa_results
    > mmmu_results
    :: repo=@
    :: eval_bsz=@
    :: tokenizer_model=$model_name
    :: prompt_format=@
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres
{
    python $repo/examples/multimodal/evaluate_mmmu.py \
        --input-path $mmmu_outputs/MMMU_outputs \
        | tee $mmmu_results

    python $repo/examples/multimodal/evaluate_textvqa.py \
        --input-path $textvqa_outputs/TextVQA_outputs \
        | tee $textvqa_results

    python $repo/examples/multimodal/evaluate_ai2d.py \
        --input-path $ai2d_outputs/AI2D_outputs \
        | tee $ai2d_results
}

func ConvertToHF
    < model_dir
    > hf_model_dir
    :: repo
    :: model_name
    :: upload_id
    :: hf_model_type
{
    python $repo/examples/multimodal/convert_to_hf.py     \
        --mcore-load-dir $model_dir  \
        --hf-save-dir $hf_model_dir   \
        --original-text-model-id $model_name     \
        --original-vision-model-id openai/clip-vit-large-patch14-336 \
        --hf-model-type $hf_model_type \
        --upload-to-hub $upload_id
}

task ConvertPretrainedModel calls ConvertToHF
    < model_dir=@PretrainModel
    > hf_model_dir
    :: repo=@
    :: model_name=@
    :: upload_id=$prt_upload_id
    :: hf_model_type=@
    :: .submitter=@
    :: .account=$convert_account
    :: .time=$convert_time
    :: .cpus=$convert_cpus
    :: .partition=$convert_partition

task ConvertFinetunedModel calls ConvertToHF
    < model_dir=$finetuned_dir@FineTuneModel
    > hf_model_dir
    :: repo=@
    :: model_name=@
    :: upload_id=$sft_upload_id
    :: hf_model_type=@
    :: .submitter=@
    :: .account=$convert_account
    :: .time=$convert_time
    :: .cpus=$convert_cpus
    :: .partition=$convert_partition

summary Evaluation {
    of EvaluatePretrainedModel > COCOCider {
        cat $coco_results | grep -o "CIDEr: [0-9.]\+" | sed "s/CIDEr: //" > $COCOCider
    }
    of EvaluateFinetunedModel > MMMUAccuracy TextVQAAccuracy AI2DAccuracy {
        cat $mmmu_results | grep -o "MMMU average accuracy: [0-9.]\+" | sed "s/MMMU average accuracy: //" > $MMMUAccuracy
        cat $textvqa_results | grep -o "TextVQA Accuracy [0-9.]\+" | sed "s/TextVQA Accuracy //" > $TextVQAAccuracy
        cat $ai2d_results | grep -o "AI2D Accuracy [0-9.]\+" | sed "s/AI2D Accuracy //" > $AI2DAccuracy
    }
}

plan TrainPipelineQwen25 {
    reach EvaluateFinetunedModel via (TextModel: qwen2p5_7b)
}

plan TrainPipelineEuroLLM {
    reach EvaluateFinetunedModel via (TextModel: eurollm_9b)
}

# plan SweepTrainQwen25 {
#     reach EvaluatePretrainedModel via (TextModel: qwen2p5_7b) * (PretrainIters: 2000 5000) * (PretrainLR: 0p001)
#     reach EvaluatePretrainedModel via (TextModel: qwen2p5_7b) * (PretrainIters: 5000) * (PretrainLR: 0p0005)
#     reach EvaluatePretrainedModel via (TextModel: qwen2p5_7b) * (FullUnfreeze: true) * (PretrainIters: 2000 5000) * (PretrainLR: 0p001 0p0005)
# }
# 
# plan TrainPipelineEuroLLM9B {
#     reach EvaluatePretrainedModel, EvaluateFinetunedModel via (TextModel: eurollm_9b)
#     reach EvaluateFinetunedModel via (TextModel: eurollm_9b) * (ImagePreproc: nvlm) * (PixelShuffle: true)
# }
# 
# plan BackboneComparison {
#     reach EvaluateFinetunedModel via (TextModel: mistral qwen2p5_7b eurollm_9b)
#     reach EvaluateFinetunedModel via (TextModel: qwen2p5_7b) * (ImagePreproc: nvlm) * (PixelShuffle: true)
# }
# 
# plan ConvertModels {
#     reach ConvertFinetunedModel via (TextModel: qwen2p5_7b) * (ImagePreproc: nvlm) * (PixelShuffle: true)
# }
# 
# plan TestPipeline {
#     reach PretrainModel via (TextModel: eurollm_9b)
# }
