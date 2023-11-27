PRE_SEQ_LEN=128
CHECKPOINT=initial_model
STEP=3000

CUDA_VISIBLE_DEVICES=0 nohup python3 main.py \
    --do_predict \
    --validation_file dev.json \
    --test_file dev.json \
    --overwrite_cache \
    --prompt_column question \
    --response_column report \
    --model_name_or_path /root/autodl-tmp/projects/ChatGLM-6B/chatglm-6b \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 >> eval_0602.out 2>&1 &
