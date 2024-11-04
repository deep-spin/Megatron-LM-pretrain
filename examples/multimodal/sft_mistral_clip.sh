#!/bin/bash
# Run SFT on a pretrained multimodal model

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_NAME="mcore-llava-mistral-7b-instruct-clip336-sft"

# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

if [[ -z $LOAD_NAME ]]; then
    echo "Please set LOAD_NAME for input model name."
    exit 1
fi

if [[ -z $TOKENIZER_MODEL ]]; then
    echo "Please set TOKENIZER_MODEL for tokenizer model name."
    exit 1
fi

CHECKPOINT_DIR="$LOAD_NAME"

DATA_TRAIN="${SOURCE}/examples/multimodal/sft_dataset.yaml"

DEBUG=0
if [[ $DEBUG -eq 1 ]]; then
    BZ=8
    NW=1
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=128
    NW=2
    HD=0.1
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi

OPTIONS=" \
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
    --num-workers ${NW} \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 576 \
    --decoder-seq-length 2048 \
    --max-position-embeddings 4096 \
    --ffn-hidden-size 14336 \
    --train-iters 20000 \
    --micro-batch-size 8 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 20000 \
    --lr-warmup-fraction .01 \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 500 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 500 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --split 100,0,0 \
    --clip-grad 0.5 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --eod-mask-loss \
    --bf16 \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 336 \
    --img-w 336 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=mistral_7b \
    --disable-vision-class-token \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60
"

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

torchrun --nproc_per_node 4 --master-port 29700 examples/multimodal/train.py ${OPTIONS}
