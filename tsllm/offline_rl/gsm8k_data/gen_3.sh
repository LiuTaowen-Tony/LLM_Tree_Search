set -e

K=100
T=0.7
N_WORKER=16
OUTPUT_DIR=./gsm8k_data/cot_sample/
CUDA_DEVICES=1,2

CT2_CACHE=$1
TOKENIZER_PATH=$2

python generate_data.py \
    -k $K \
    -t $T \
    --num_workers $N_WORKER \
    --gpu_ids $CUDA_DEVICES \
    --ct2_dir ${CT2_CACHE}/llama2_sft_ep1_ct2 \
    --tokenizer_path $TOKENIZER_PATH \
    --output_path ${OUTPUT_DIR}/gsm8k_train_cot_sample_offline_sft_k${K}_ep1.jsonl \
    --env_name gsm8k

# python generate_data.py \
#     -k $K \
#     -t $T \
#     --num_workers $N_WORKER \
#     --gpu_ids $CUDA_DEVICES \
#     --ct2_dir ${CT2_CACHE}/llama2_sft_ep2_ct2 \
#     --tokenizer_path $TOKENIZER_PATH \
#     --output_path ${OUTPUT_DIR}/gsm8k_train_cot_sample_offline_sft_k${K}_ep2.jsonl \
#     --env_name gsm8k


# python generate_data.py \
#     -k $K \
#     -t $T \
#     --num_workers $N_WORKER \
#     --gpu_ids $CUDA_DEVICES \
#     --ct2_dir ${CT2_CACHE}/llama2_sft_ep3_ct2 \
#     --tokenizer_path $TOKENIZER_PATH \
#     --output_path ${OUTPUT_DIR}/gsm8k_train_cot_sample_offline_sft_k${K}_ep3.jsonl \
#     --env_name gsm8k