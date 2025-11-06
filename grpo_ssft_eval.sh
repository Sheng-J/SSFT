export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCH_COMPILE=1
export TORCHINDUCTOR_CACHE_DIR=temp_cache3 

MODEL=shengjia-toronto/grpo-test-ssft-32B
OUTPUT_DIR=data/evals/$MODEL
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=8,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:100,temperature:0.7,top_p:0.95}"


TASK=aime25_exp
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

# Pass@1 eval
TASK=aime24_exp
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


