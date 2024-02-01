# Note that For RLHF we do not conduct SFT training, we directly utilize vicgalle/gpt2-open-instruct-v1.
TOKENIZERS_PARALLELISM=false accelerate launch --config_file mcts_gsm8k_llama_deepspeed.yaml train_gsm8k_sft.py 

# Critic training for all four tasks, data is collected by data collection section.
TOKENIZERS_PARALLELISM=false accelerate launch --config_file mcts_gsm8k_llama_deepspeed.yaml train_gsm8k_critic.py
