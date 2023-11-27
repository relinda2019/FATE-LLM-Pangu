PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 nohup python3 main.py \
    --do_train \
    --train_file train.json \
    --validation_file dev.json \
    --prompt_column question \
    --response_column report \
    --overwrite_cache \
    --model_name_or_path /root/autodl-tmp/projects/ChatGLM-6B/chatglm-6b \
    --output_dir output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 >> train_0522.out 2>&1 &

