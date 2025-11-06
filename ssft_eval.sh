export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export VLLM_DISABLE_COMPILE_CACHE=1
export TORCH_COMPILE=1
export TORCHINDUCTOR_CACHE_DIR=temp_cache3 

MODEL=shengjia-toronto/ssft-32B-N6
OUTPUT_DIR=data/evals/$MODEL
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=8,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.7,top_p:0.95}"

# Uncomment for run the corresponding evaluations

# Cons@6 Cons@32 eval
TASK=aime25_sot6
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

# Pass@1 eval
TASK=aime25_greedythink1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


TASK=aime24_sot6
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


TASK=aime24_greedythink1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


TASK=math_500_sot6
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


TASK=math_500_greedythink1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


TASK=gpqa_diamond_instruct_lighteval_sot6
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details


TASK=gpqa_diamond_instruct_lighteval_greedythink1
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

