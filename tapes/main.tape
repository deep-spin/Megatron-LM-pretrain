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
    mistral_model=(
        TextModel:
            mistral="mistralai/Mistral-7B-Instruct-v0.3"
            tower="Unbabel/TowerInstruct-Mistral-7B-v0.2"
    )
    prompt_format=(
        TextModel:
            mistral="mistral"
            tower="chatml"
    )
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

    # fine-tuning arguments
    finetune_model_dir=/lustre/fswork/projects/rech/qjm/ued79zb/towervision_ckpts/mistral_7b_instruct_sft
    finetune_tensorboard=/lustre/fswork/projects/rech/qjm/ued79zb/towervision_tbs/mistral_7b_instruct_sft
    finetune_iters=5000
    finetune_bsz=128
    finetune_lr=1e-6
    finetune_lr_warmup=0.01
    finetune_save_interval=1000
    finetune_eval_interval=1000

    # eval arguments
    coco_dir=/lustre/fswork/projects/rech/qjm/ued79zb/coco/
    mmmu_dir=/dev/null
    eval_bsz=32

    # convert arguments
    upload_id="patricksf/mistral-7b-clip-prt"

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

    finetune_C="h100"
    finetune_account="qjm@h100"
    finetune_time="10:00:00"
    finetune_cpus=80
    finetune_gres="gpu:4"

    eval_C="h100"
    eval_account="qjm@h100"
    eval_time="1:00:00"
    eval_cpus=80
    eval_gres="gpu:4"

    convert_account="qjm@cpu"
    convert_time="1:00:00"
    convert_cpus=4
    convert_partition="prepost"
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
    :: tokenizer_model=$mistral_model
    :: prompt_format=@
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
        --tokenizer-type MultimodalTokenizer \
        --tokenizer-model ${tokenizer_model} \
        --tokenizer-prompt-format ${prompt_format} \
        --data-path ${repo}/examples/multimodal/pretrain_dataset.yaml \
        --prompt-path ${repo}/examples/multimodal/manual_prompts.json \
        --save-interval ${save_interval} \
        --save ${model_dir} \
        --load ${model_dir} \
        --dataloader-save ${model_dir}/dataloader \
        --pretrained-checkpoint ${initial_model} \
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
        --allow-missing-vision-projection-checkpoint \
        --ckpt-format torch
}


task EvaluatePretrainedModel
    < coco_dir=@
    < model_dir=@PretrainModel
    > coco_results
    :: repo=@
    :: eval_bsz=@
    :: tokenizer_model=$mistral_model
    :: prompt_format=@
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres
{
    echo "Evaluating pretrained model..."
    # TODO: inline the bash script
    bash $repo/examples/multimodal/text_generation_mistral_clip.sh \
        --input-image-path $coco_dir \
        --gt-path $coco_dir/coco_karpathy_test.json \
        --output-path coco_outputs \
        --model-path $model_dir \
        --tokenizer-path $tokenizer_model \
        --tokenizer-prompt-format ${prompt_format} \
        --batch-size $eval_bsz \
        --task captioning
    python $repo/examples/multimodal/evaluate_coco.py \
        --input-path coco_outputs \
        --groundtruth-path $coco_dir/coco_karpathy_test_gt.json \
        | tee $coco_results
}

task FineTuneModel
    < pretrained_dir=$model_dir@PretrainModel
    > finetuned_dir
    :: repo=@
    :: tokenizer_model=$mistral_model
    :: prompt_format=@
    :: train_iters=$finetune_iters
    :: batch_size=$finetune_bsz
    :: lr=$finetune_lr
    :: lr_warmup_fraction=$finetune_lr_warmup
    :: save_interval=$finetune_save_interval
    :: eval_interval=$finetune_eval_interval
    :: external_resume=true
    :: external_model_dir=$finetune_model_dir
    :: external_tensorboard=$finetune_tensorboard
    :: .submitter=@
    :: .C=$finetune_C
    :: .account=$finetune_account
    :: .time=$finetune_time
    :: .cpus=$finetune_cpus
    :: .gres=$finetune_gres
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
        --decoder-seq-length 2048 \
        --max-position-embeddings 4096 \
        --ffn-hidden-size 14336 \
        --train-iters ${train_iters} \
        --micro-batch-size 8 \
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
        --data-path ${repo}/examples/multimodal/sft_dataset.yaml \
        --prompt-path ${repo}/examples/multimodal/manual_prompts.json \
        --save-interval ${save_interval} \
        --save ${finetuned_dir} \
        --load ${finetuned_dir} \
        --dataloader-save ${finetuned_dir}/dataloader \
        --pretrained-checkpoint ${pretrained_dir} \
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
        --freeze-ViT \
        --patch-dim 14 \
        --img-h 336 \
        --img-w 336 \
        --dataloader-type external \
        --tensorboard-dir ${tensorboard} \
        --language-model-type=mistral_7b \
        --disable-vision-class-token \
        --distributed-timeout-minutes 60 \
        --ckpt-format torch
}

task EvaluateFinetunedModel
    < mmmu_dir=@
    < finetuned_dir=@FineTuneModel
    > mmmu_results
    :: repo=@
    :: eval_bsz=@
    :: tokenizer_model=$mistral_model
    :: prompt_format=@
    :: .submitter=@
    :: .C=$eval_C
    :: .account=$eval_account
    :: .time=$eval_time
    :: .cpus=$eval_cpus
    :: .gres=$eval_gres
{
    export PYTHONPATH="$PYTHONPATH:$repo/examples/multimodal"
    bash $repo/examples/multimodal/text_generation_mistral_clip.sh \
        --input-image-path none \
        --output-path mmmu_outputs \
        --model-path $finetuned_dir \
        --tokenizer-path $tokenizer_model \
        --tokenizer-prompt-format ${prompt_format} \
        --batch-size $eval_bsz \
        --task MMMU
    python $repo/examples/multimodal/evaluate_mmmu.py \
        --input-path mmmu_outputs \
        | tee $mmmu_results
}

func ConvertToHF
    < model_dir
    > hf_model_dir
    :: repo
    :: mistral_model
    :: upload_id
{
    python $repo/examples/multimodal/convert_to_hf.py     \
        --mcore-load-dir $model_dir  \
        --hf-save-dir $hf_model_dir   \
        --original-text-model-id $mistral_model     \
        --original-vision-model-id openai/clip-vit-large-patch14-336 \
        --upload-to-hub $upload_id
}

task ConvertPretrainedModel calls ConvertToHF
    < model_dir=@PretrainModel
    > hf_model_dir
    :: repo=@
    :: mistral_model=@
    :: upload_id=@
    :: .submitter=@
    :: .account=$convert_account
    :: .time=$convert_time
    :: .cpus=$convert_cpus
    :: .partition=$convert_partition

task ConvertFinetunedModel calls ConvertToHF
    < model_dir=$finetuned_dir@FineTuneModel
    > hf_model_dir
    :: repo=@
    :: mistral_model=@
    :: .submitter=@
    :: .account=$convert_account
    :: .time=$convert_time
    :: .cpus=$convert_cpus
    :: .partition=$convert_partition


plan TrainPipelineMistral {
    reach EvaluatePretrainedModel
    reach EvaluateFinetunedModel
}

plan TrainPipelineTower {
    reach EvaluatePretrainedModel via (TextModel: tower)
    reach EvaluateFinetunedModel via (TextModel: tower)
}

plan ConvertModels {
    reach ConvertPretrainedModel
    #reach ConvertFinetunedModel
}
