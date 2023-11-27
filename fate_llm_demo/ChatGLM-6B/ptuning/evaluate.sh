PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-2e-2
STEP=3000

CUDA_VISIBLE_DEVICES=0 nohup python3 main.py \
    --do_predict \
    --validation_file dev.json \
    --test_file dev.json \
    --overwrite_cache \
    --prompt_column question \
    --response_column report \
    --model_name_or_path /data/standalone_fate_install_1.11.3_release/fate_llm_demo/ChatGLM-6B/chatglm-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 >> eval_1126.out 2>&1 &
