import "scslurm.tape"

global {
    ducttape_experimental_imports=true
    ducttape_experimental_submitters=true
    ducttape_experimental_multiproc=true

    ducttape_output=/lustre/fswork/projects/rech/qjm/ued79zb/towervision_outputs
    repo=/linkhome/rech/genrce01/ued79zb/repos/Megatron-LM-pretrain

    # multimodal model parameters
    # (base lm, tp, etc...)
    clip_original_dir=/lustre/fswork/projects/rech/qjm/ued79zb/clip_model_og/
    mistral_model="mistralai/Mistral-7B-Instruct-v0.3"
    tp=4
    pp=1

    # pre-training arguments
    external_model_dir=/lustre/fswork/projects/rech/qjm/ued79zb/towervision_ckpts/mistral_7b_instruct_prt
    external_tensorboard=/lustre/fswork/projects/rech/qjm/ued79zb/towervision_tbs/mistral_7b_instruct_prt
    pretrain_iters=2000
    pretrain_bsz=256
    pretrain_lr=0.001
    pretrain_lr_warmup=0.03
    pretrain_save_interval=500
    pretrain_eval_interval=500


    # -- submitter arguments --
    submitter=scslurm

    prepare_account="qjm@cpu"
    prepare_time="1:00:00"
    prepare_cpus=4
    prepare_partition="prepost"

    pretrain_C="h100"
    pretrain_account="qjm@h100"
    pretrain_time="10:00:00"
    pretrain_cpus=80
    pretrain_gres="gpu:4"
}

task PrepareModel
    > initial_model
    :: repo=@
    :: clip_original_dir=@
    :: mistral_model=@
    :: tp=@
    :: pp=@
    :: .submitter=@
    :: .account=$prepare_account
    :: .time=$prepare_time
    :: .cpus=$prepare_cpus
    :: .partition=$prepare_partition
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
        --model ${mistral_model} \
        --output-dir mistral_original_dir
    python $repo/tools/checkpoint/convert.py --model-type GPT \
        --loader llama_mistral \
        --saver mcore \
        --checkpoint-type hf \
        --model-size mistral-7B \
        --load-dir mistral_original_dir \
        --save-dir mistral_mcore_dir \
        --tokenizer-model ${mistral_model} \
        --target-tensor-parallel-size ${tp} \
        --target-pipeline-parallel-size ${pp} \
        --bf16

    # Combine models
    echo "Combining language and vision models..."
    bash $repo/examples/multimodal/combine_mistral_clip.sh \
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
    :: tokenizer_model=$mistral_model
    :: train_iters=$pretrain_iters
    :: batch_size=$pretrain_bsz
    :: lr=$pretrain_lr
    :: lr_warmup_fraction=$pretrain_lr_warmup
    :: save_interval=$pretrain_save_interval
    :: eval_interval=$pretrain_eval_interval
    :: external_resume=true
    :: external_model_dir=@
    :: external_tensorboard=@
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

    torchrun --nproc_per_node 4 $repo/examples/multimodal/train.py \
        --apply-layernorm-1p \
        --attention-softmax-in-fp32 \
        --use-checkpoint-args \
        --use-distributed-optimizer \
        --transformer-impl transformer_engine \
        --use-te \
        --normalization RMSNorm \
        --group-query-attention \
        --num-query-groups 8 \
        --no-masked-softmax-fusion \
        --num-workers 2 \
        --use-flash-attn \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.1 \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 1 \
        --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 576 \
        --decoder-seq-length 1024 \
        --max-position-embeddings 4096 \
        --ffn-hidden-size 14336 \
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
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${tokenizer_model} \
        --data-path ${repo}/examples/multimodal/pretrain_dataset.yaml \
        --prompt-path ${repo}/examples/multimodal/manual_prompts.json \
        --save-interval ${save_interval} \
        --save ${model_dir} \
        --load ${model_dir} \
        --dataloader-save ${model_dir}/dataloader \
        --pretrained-checkpoint ${initial_model}/mistral_instruct_clip336_tp4_combined_mcore \
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
        --freeze-LM \
        --freeze-ViT \
        --patch-dim 14 \
        --img-h 336 \
        --img-w 336 \
        --dataloader-type external \
        --tensorboard-dir ${tensorboard} \
        --language-model-type=mistral_7b \
        --disable-vision-class-token \
        --distributed-timeout-minutes 60 \
        --allow-missing-vision-projection-checkpoint
}

plan TrainLLaVA {
    reach PretrainModel
}