# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import asyncio
import gc
import itertools
import logging
import os
from typing import Coroutine, Optional
import random

import torch
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    GenerativeResponseProbDist,
)
from lighteval.models.utils import ModelConfig, _simplify_name
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
)
from lighteval.utils.imports import is_vllm_available
from lighteval.utils.utils import as_list
import numpy as np


logger = logging.getLogger(__name__)


if is_vllm_available():
    import ray
    from more_itertools import distribute
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.transformers_utils.tokenizer import get_tokenizer
    from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM

    logging.getLogger("vllm").propagate = True
    logging.getLogger("vllm").handlers.clear()

    logging.getLogger("ray").propagate = True
    logging.getLogger("ray").handlers.clear()
else:
    LLM = None
    AsyncLLM = None
    SamplingParams = None
    AsyncEngineArgs = None
    get_tokenizer = None
    ray = None
    distribute = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class VLLMModelConfig(ModelConfig):
    model_name: str
    revision: str = "main"  # revision of the model
    dtype: str = "bfloat16"
    tensor_parallel_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
    data_parallel_size: PositiveInt = 1  # how many GPUs to use for data parallelism
    pipeline_parallel_size: PositiveInt = 1  # how many GPUs to use for pipeline parallelism
    gpu_memory_utilization: NonNegativeFloat = 0.9  # lower this if you are running out of memory
    max_model_length: PositiveInt | None = (
        None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    )
    quantization: str | None = None
    load_format: str | None = None
    swap_space: PositiveInt = 4  # CPU swap space size (GiB) per GPU.
    seed: NonNegativeInt = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
    max_num_seqs: PositiveInt = 128  # maximum number of sequences per iteration; This variable and `max_num_batched_tokens` effectively control the batch size at prefill stage. See https://github.com/vllm-project/vllm/issues/2492 for detailed explaination.
    max_num_batched_tokens: PositiveInt = 2048  # maximum number of tokens per batch
    subfolder: str | None = None
    is_async: bool = False  # Whether to use the async version or sync version of the model


class VLLMModel(LightevalModel):
    def __init__(
        self,
        config: VLLMModelConfig,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation."""
        self._config = config
        self.use_chat_template = config.use_chat_template
        self.data_parallel_size = config.data_parallel_size
        self.tensor_parallel_size = config.tensor_parallel_size
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False
        self._tokenizer = self._create_auto_tokenizer(config)

        # NOTE SJ: max_model_length is copied to _max_length here
        # max_model_length doesn't matter anymore
        self._max_length = config.max_model_length if config.max_model_length is not None else None

        # If model_parallel is not set we compare the number of processes with the number of GPUs
        self.model = self._create_auto_model(config)

        # self._device = config.accelerator.device if config.accelerator is not None else "cpu"
        self.multichoice_continuations_start_space = config.multichoice_continuations_start_space

        self.model_name = _simplify_name(config.model_name)
        self.model_sha = ""
        self.precision = config.dtype

        self.model_info = ModelInfo(model_name=self.model_name, model_sha=self.model_sha)
        self.pairwise_tokenization = config.pairwise_tokenization

    @property
    def tokenizer(self):
        return self._tokenizer

    def cleanup(self):
        destroy_model_parallel()
        if self.model is not None:
            del self.model
        gc.collect()
        ray.shutdown()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def max_length(self) -> int:
        return self._max_length

    def _create_auto_model(self, config: VLLMModelConfig) -> Optional[LLM]:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
        }

        if config.quantization is not None:
            self.model_args["quantization"] = config.quantization
        if config.load_format is not None:
            self.model_args["load_format"] = config.load_format

        if config.data_parallel_size > 1:
            self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"
            return None

        model = LLM(**self.model_args)

        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model

    def _create_auto_tokenizer(self, config: VLLMModelConfig):
        tokenizer = get_tokenizer(
            config.model_name,
            tokenizer_mode="auto",
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.
            override_bs (int, optional): Override the batch size for generation. Defaults to None.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        #NOTE SJ len(requests)=500 for math_500, len(requests)=30 for aime24
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        results = []
        # prompt_logprobs_k = 0
        prompt_logprobs_k = None

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=False,  # self.disable_tqdm,
        ):
            # For chat models, generation stops with EOS token, so we don't need to specify stop tokens
            if self.use_chat_template:
                stop_tokens = []
            else:
                # NOTE: we are assuming all items in a batch behave similarly (same
                # stop_tokens and max_tokens genrated) which is not necessarily
                # the case! Because of that we only use batch size of 1
                stop_tokens = split[0].stop_sequence

            max_new_tokens = self._config.generation_parameters.max_new_tokens or split[0].generation_size
            returns_logits = split[0].use_logits
            num_samples = split[0].num_samples
            # import ipdb; ipdb.set_trace()

            context = [sample.context for sample in split]
            # import ipdb; ipdb.set_trace()  
            # TODO 1: check if use_logits is set to True, can we reuse this to indicate debug sot think mode
            # TODO 2: (1) question-level <think i> distribution, compute logprobs for each prompt_logprobs under <think i>, computed individually
            #         (1.5) one shot compute the <think i> distribution(?) by keeping top 6 logprobs at each location, only need to generate one per question
            #         (2) compute avg of subsequent completion losses prompted with a distinct <think i>
            ###### NOTE SJ: branch to SoT mode
            if "greedythink_samplerest" in requests[0].task_name:
                print(f"Greedy Thinking then sample rest")
                # tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)
                # inputs = tokenized["input_ids"]
                vllm_outputs = self._generate_greedythink(
                    contexts=context,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=prompt_logprobs_k,
                    num_samples=num_samples,
                    generate=True
                )
            elif "greedythink" in requests[0].task_name:
                think_id = int(requests[0].task_name[-3])
                # think_id = int(requests[0].task_name[-1]) incorrect
                postfix = f"<|im_start|><think{think_id}>\n"
                context = [ctx+postfix for ctx in context]
                print(f"Greedy Thinking with postfix={postfix}")
                tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)
                inputs = tokenized["input_ids"]
                context_size = len(inputs[0])
                # assert self.max_length == max_new_tokens
                vllm_outputs = self._generate(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    stop_tokens=stop_tokens,
                    returns_logits=None,
                    num_samples=num_samples,
                    prompt_logprobs_k=prompt_logprobs_k,
                )
            elif hasattr(requests[0], 'num_sot_tags') and (requests[0].num_sot_tags is not None) and requests[0].num_sot_tags>0:
                if "1sample" in requests[0].task_name:
                    prompt_logprobs_k = 0
                num_sot_tags = requests[0].num_sot_tags
                def _fair_split_counts(total: int, parts: int) -> list[int]:
                    # e.g. total=10, parts=3 -> [4,3,3]
                    base, rem = divmod(total, parts)
                    return [base + (1 if i < rem else 0) for i in range(parts)]

                # base_tok = self.tokenizer(context, add_special_tokens=self.add_special_tokens)
                # base_inputs = base_tok["input_ids"]
                # base_ctx_len = len(base_inputs[0])
                # 1) build suffixes and a fair split of num_samples
                if num_samples < num_sot_tags:
                    # pick tag indices in 1..num_sot_tags (sorted for stable order)
                    chosen = sorted(random.sample(range(1, num_sot_tags + 1), k=num_samples))
                    suffixes = [f"<|im_start|><think{i}>\n" for i in chosen]
                    split_counts = _fair_split_counts(num_samples, len(suffixes))  # effectively [1]*num_samples
                else:
                    suffixes = [f"<|im_start|><think{i+1}>\n" for i in range(num_sot_tags)]
                    split_counts = _fair_split_counts(num_samples, num_sot_tags)
                merged_outputs = None  # final list[RequestOutput]-like
                print("\n\n-----------------------------------------------------------")
                print(f"Chosen think tags: {suffixes}")
                print(f"Split counts: {split_counts}")
                print("----------------------------------------------------------------\n\n")
                # import ipdb #NOTE num_samples=4, 
                # import torch.distributed as dist
                # port = 4444+dist.get_rank()
                # print(f"Process {dist.get_rank()} waiting for debugger on port {port}")
                # ipdb.set_trace()
                all_suffix_ids_probs = [[] for _ in range(len(context))]
                for suffix, n_this in zip(suffixes, split_counts):
                    print(f"Generating with {suffix} appended, with n_samples_this={n_this}")
                    if n_this == 0:
                        continue

                    # 2) tag prompts for this bucket and tokenize+truncate
                    tagged_context = [ctx + suffix for ctx in context]
                    tok = self.tokenizer(tagged_context, add_special_tokens=self.add_special_tokens)
                    inputs = tok["input_ids"]
                    ctx_len = len(inputs[0])
                    ##### consistency check #####
                    prompt_ids_ = self.tokenizer(context, add_special_tokens=self.add_special_tokens)['input_ids']
                    suffix_ids_ = self.tokenizer(suffix, add_special_tokens=self.add_special_tokens)['input_ids']
                    for prompt_ids_mb_, inputs_ in zip(prompt_ids_, inputs):
                        assert len(prompt_ids_mb_)+len(suffix_ids_) == len(inputs_)
                    #############################
                    # import ipdb; ipdb.set_trace()

                    # 3) generate only this bucket's share
                    # outs_t = self._generate(
                    #     inputs=inputs,
                    #     max_new_tokens=max_new_tokens,
                    #     stop_tokens=stop_tokens,
                    #     returns_logits=returns_logits,
                    #     num_samples=n_this,
                    #     prompt_logprobs_k=prompt_logprobs_k,
                    # )
                    outs_t = self._generate(
                        inputs=inputs,
                        max_new_tokens=max_new_tokens,
                        stop_tokens=stop_tokens,
                        returns_logits=prompt_logprobs_k,
                        num_samples=n_this,
                        prompt_logprobs_k=prompt_logprobs_k,
                    )
                    if prompt_logprobs_k is not None:
                        for i, outs_t_questioni in enumerate(outs_t):
                            think_id_logprob_sum = 0
                            # import ipdb;ipdb.set_trace()
                            think_logprobs = outs_t_questioni.prompt_logprobs[-len(suffix_ids_):]
                            for s_id, t_logprob in zip(suffix_ids_, think_logprobs):
                                think_id_logprob_sum += t_logprob[s_id].logprob
                            all_suffix_ids_probs[i].append(think_id_logprob_sum)

                    # import ipdb; ipdb.set_trace()
                    # self.tokenizer.decode(outs_t[0].prompt_token_ids)=
                    # "....<|im_start|>assistant\n<|im_start|><think1>\n"
                    #NOTE len(outs_t[0].outputs)=11,  outs_t[0].
                    # 4) merge: concatenate per-question .outputs
                    if merged_outputs is None:
                        merged_outputs = outs_t
                        # normalize prompt_token_ids to the *base* tokens (no tag)
                        # for i in range(len(merged_outputs)):
                        #     merged_outputs[i].prompt_token_ids = base_inputs[i]
                    else:
                        # len(merged_outputs)=30 questions
                        for i in range(len(merged_outputs)):
                            # all merged_outputs[i] already store prompt_logprobs
                            merged_outputs[i].outputs.extend(outs_t[i].outputs)

                # 5) (safety) each question should end with exactly num_samples generations
                for i in range(len(merged_outputs)):
                    assert len(merged_outputs[i].outputs) == num_samples, \
                        f"Question {i} has {len(merged_outputs[i].outputs)} samples, expected {num_samples}"

                vllm_outputs = merged_outputs
            else:
                print("\ndefault generate\n")
                tokenized = self.tokenizer(context, add_special_tokens=self.add_special_tokens)

                # The main question for this step is the following:
                # Would we rather truncate the prompt to allow generation to go to max_new_tokens, at the risk
                # of losing some meaning, or have some generations that are exceedingly short?
                # The choice we go for here is to avoid truncating the prompt if we can, since it
                # should have been managed by the prompt creator/few shot manager if requested by the user.
                inputs = tokenized["input_ids"]
                context_size = len(inputs[0])

                # left truncate the inputs to the maximum length
                # NOTE 
                # max_new_tokens=32768, max_model_length=32768
                # if max_new_tokens is not None:
                #     if context_size + max_new_tokens > self.max_length:
                #         logger.warning(
                #             f"{context_size + max_new_tokens=} which is greater than {self.max_length=}. Truncating context to {self.max_length - max_new_tokens} tokens."
                #         )
                #         context_size = self.max_length - max_new_tokens
                #         if context_size < 0:
                #             logger.critical(
                #                 f"{context_size=} is less than 0, either reduce the max_new_tokens or increase model max length."
                #             )
                #             raise ValueError("Context size is less than 0.")
                #         inputs = [input[-context_size:] for input in inputs]
                # else:
                #     if context_size > self.max_length:
                #         logger.warning(
                #             f"{context_size=} which is greater than {self.max_length=}. Truncating context to {self.max_length} tokens."
                #         )
                #         context_size = self.max_length
                #         inputs = [input[-context_size:] for input in inputs]

                # vllm_outputs = self._generate(
                #     inputs=inputs,
                #     max_new_tokens=max_new_tokens,
                #     stop_tokens=stop_tokens,
                #     returns_logits=returns_logits,
                #     num_samples=num_samples,
                # )
                vllm_outputs = self._generate(
                        inputs=inputs,
                        max_new_tokens=max_new_tokens,
                        stop_tokens=stop_tokens,
                        returns_logits=prompt_logprobs_k,
                        num_samples=num_samples,
                        prompt_logprobs_k=prompt_logprobs_k,
                    )

            # TODO (2) compute avg of subsequent completion losses prompted with a distinct <think i> (for debug now)
            for question_id, vllm_output in enumerate(vllm_outputs): # len(vllm_outputs)=30
                # import ipdb; ipdb.set_trace()
                output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
                # len(logprobs)=0
                logprobs = [output.logprobs for output in vllm_output.outputs] or []
                result = [output.text for output in vllm_output.outputs]
                input_token_ids = vllm_output.prompt_token_ids

                per_completion_neglogprob_losses = []
                per_1k_completion_neglogprob_losses = []

                per_sotprompt_completion_neglogprob_losses = []
                per_1k_sotprompt_completion_neglogprob_losses = []

                if prompt_logprobs_k is not None and hasattr(requests[0], 'num_sot_tags') and (requests[0].num_sot_tags is not None) and requests[0].num_sot_tags>0:
                    logprobs_backup = [logprob[token_id].logprob for token_id, logprob in zip(output_token_ids[0], logprobs[0])]
                    sanity_check_per_think_probs_ = all_suffix_ids_probs[question_id]
                    sanity_check_per_think_probs = np.exp(sanity_check_per_think_probs_).tolist()
                    assert len(sanity_check_per_think_probs) == num_sot_tags
                    split_ = len(logprobs) // num_sot_tags
                    for pred_idx, (logprobs_, output_token_ids_) in enumerate(zip(logprobs, output_token_ids)):
                        sot_id = pred_idx // split_
                        logprobs_cleaned_ = [logprob_[token_id_].logprob for token_id_, logprob_ in zip(output_token_ids_, logprobs_)]
                        per_completion_neglogprob_losses.append(-np.mean(logprobs_cleaned_).item())
                        per_1k_completion_neglogprob_losses.append(-np.mean(logprobs_cleaned_[:1000]).item())
                        # sot_loss = -sum(sanity_check_per_think_probs_)
                        # cat1 = sanity_check_per_think_probs_ + logprobs_cleaned_
                        # cat2 = sanity_check_per_think_probs_ + logprobs_cleaned_[:1000]
                        # per_sotprompt_completion_neglogprob_losses.append(-np.mean(cat1).item())
                        # per_1k_sotprompt_completion_neglogprob_losses.append(-np.mean(cat2).item())
                        per_sotprompt_completion_neglogprob_losses.append(None)
                        per_1k_sotprompt_completion_neglogprob_losses.append(None)
                        # import ipdb; ipdb.set_trace() # len(vllm_output.prompt_logprobs)=873
                        # print()
                else:
                    logprobs_backup = []
                    # sanity_check_per_think_probs = [None for _ in range(num_sot_tags)]
                    sanity_check_per_think_probs = [None]
                # import ipdb; ipdb.set_trace()

                # import ipdb; ipdb.set_trace()

                # import ipdb; ipdb.set_trace()
                # cur_response = GenerativeResponse(
                #     result=result,
                #     logits=logprobs,
                #     generated_tokens=list(output_token_ids),
                #     input_tokens=input_token_ids,
                # )
                cur_response = GenerativeResponseProbDist(
                    result=result,
                    logits=logprobs_backup,
                    generated_tokens=list(output_token_ids),
                    input_tokens=input_token_ids,
                    sanity_check_per_think_probs=sanity_check_per_think_probs, # [6]
                    per_completion_neglogprob_losses=per_completion_neglogprob_losses, #[66]
                    per_sotprompt_completion_neglogprob_losses=per_sotprompt_completion_neglogprob_losses,
                    per_1k_completion_neglogprob_losses=per_1k_completion_neglogprob_losses,
                    per_1k_sotprompt_completion_neglogprob_losses=per_1k_sotprompt_completion_neglogprob_losses,
                )
                results.append(cur_response)
        # NOTE SJ: add a breakpoint here
        return dataset.get_original_order(results)

    def _generate_greedythink(
        self,
        contexts: list[str],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
        *,
        max_greedy_steps: int = 256,
        tail_window_tokens: int = 32,
    ) -> list[GenerativeResponse]:
        """
        Stage A: Greedy (batched) 1-token stepping until a <thinki> tag appears for each prompt.
                 (Detection is done via decoding and string matching.)
        Stage B: Append the FOUND STRING to the original context, re-tokenize, then call _generate(...) once.
        Fails fast with a simple assert if any prompt does not surface a <thinki> within `max_greedy_steps`.
        """
        import re
        from more_itertools import distribute

        # 1) Tokenize original contexts once for Stage A
        tok = self.tokenizer(contexts, add_special_tokens=self.add_special_tokens)
        inputs: list[list[int]] = tok["input_ids"]

        THINK_TAG_RE = re.compile(r"<think(\d+)>")

        def _tail_has_think_tag(tokenizer, token_ids):
            tail = token_ids[-tail_window_tokens:] if len(token_ids) > tail_window_tokens else token_ids
            txt = tokenizer.decode(tail, skip_special_tokens=False)
            return THINK_TAG_RE.search(txt) is not None

        # --- Stage-A per-shard greedy runner returns prefix TEXTS (not ids) ---
        def _greedy_stage_local(model_args: dict, shard_inputs: list[list[int]]) -> list[str]:
            from vllm import LLM, SamplingParams
            # NOTE this worked!!! test just return "<|im_start|><think2>\n"
            # return ["<|im_start|><think2>\n" for _ in shard_inputs]

            llm = LLM(**model_args)
            tokenizer = llm.get_tokenizer()

            greedy_sp = SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1,
                n=1, max_tokens=1, stop=None,
                detokenize=False, logprobs=None
            )

            grown_prefixes: list[list[int]] = [[] for _ in shard_inputs]
            found_mask = [False] * len(shard_inputs)

            active_idx = list(range(len(shard_inputs)))
            step = 0
            while active_idx and step < max_greedy_steps:
                batch = [shard_inputs[i] + grown_prefixes[i] for i in active_idx]
                outs = llm.generate(prompt_token_ids=batch, sampling_params=greedy_sp)

                # add one new token for each active item
                for pos, out in enumerate(outs):
                    i = active_idx[pos]
                    if out.outputs and out.outputs[0].token_ids:
                        grown_prefixes[i].append(out.outputs[0].token_ids[-1])

                # check tag presence
                still_active = []
                for i in active_idx:
                    if _tail_has_think_tag(tokenizer, shard_inputs[i] + grown_prefixes[i]):
                        found_mask[i] = True
                    else:
                        still_active.append(i)
                active_idx = still_active
                step += 1

            # Simple assert if any prompt failed
            assert all(found_mask), "greedy stage did not find a <thinki> tag for at least one prompt"

            # NOTE this worked!!! test just return "<|im_start|><think2>\n"
            res = [tokenizer.decode(pref, skip_special_tokens=False) for pref in grown_prefixes]
            print("found greedy think tags=", res)
            # return ["<|im_start|><think2>\n" for _ in shard_inputs]
            return res
            # Return prefix TEXTS (decode with specials kept)
            # return [tokenizer.decode(pref, skip_special_tokens=False) for pref in grown_prefixes]

        # --- Stage-A across DP shards (if any) ---
        if self.data_parallel_size > 1:
            @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def _greedy_stage_remote(model_args: dict, shard_inputs: list[list[int]]):
                return _greedy_stage_local(model_args, shard_inputs)

            shards_inputs = [list(x) for x in distribute(self.data_parallel_size, inputs)]
            obj_refs = [_greedy_stage_remote.remote(self.model_args, shard) for shard in shards_inputs]
            shard_prefix_texts = ray.get(obj_refs)  # list[list[str]]
            # Interleave shard results back to original order
            prefix_texts: list[str] = [
                x for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(s) for s in shard_prefix_texts])) if x is not None
            ]
            ray.shutdown()
        else:
            prefix_texts = _greedy_stage_local(self.model_args, inputs)

        # import ipdb; ipdb.set_trace()
        # 2) Append FOUND STRINGS to original contexts and re-tokenize (consistency)
        augmented_contexts = [ctx + pref for ctx, pref in zip(contexts, prefix_texts)]
        tok2 = self.tokenizer(augmented_contexts, add_special_tokens=self.add_special_tokens)
        stage_b_inputs: list[list[int]] = tok2["input_ids"]

        # 3) Delegate Stage-B to the standard _generate (one batched call)
        return self._generate(
            inputs=stage_b_inputs,
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            returns_logits=None,
            num_samples=num_samples,
            generate=generate,
            prompt_logprobs_k=None,
        )


    def _generate_greedythink_bad(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
    ):
        """Data-parallel inference:
           Stage-A: greedy 1-token steps until `<thinki>` appears
           Stage-B: resume with original SamplingParams.

           If `<thinki>` is NOT found within max_greedy_steps, raise RuntimeError.
        """
        # --- Stage-B (normal) sampling params ---
        sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())
        sampling_params.n = num_samples
        sampling_params.max_tokens = max_new_tokens
        sampling_params.stop = stop_tokens
        # sampling_params.logprobs = 1 if returns_logits else 0
        sampling_params.logprobs = None
        sampling_params.prompt_logprobs = None
        sampling_params.detokenize = True

        assert self.data_parallel_size >= 1
        import re
        from copy import deepcopy

        THINK_TAG_RE = re.compile(r"<think(\d+)>")

        @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
        def run_inference_one_model_remote(model_args: dict,
                                           sampling_params: SamplingParams,
                                           requests: list[list[int]],
                                           dp_rank: int):

            THINK_TAG_RE = re.compile(r"<think(\d+)>")

            def _tail_has_think_tag(tokenizer, token_ids, max_tail_tokens: int = 24):
                tail_ids = token_ids[-max_tail_tokens:] if len(token_ids) > max_tail_tokens else token_ids
                tail_text = tokenizer.decode(tail_ids, skip_special_tokens=False)
                m = THINK_TAG_RE.search(tail_text)
                return (m is not None, (m.group(0) if m else None), (m.group(1) if m else None), tail_text)

            def generate_until_think_then_sample(
                model_args: dict,
                prompt_token_ids_batch: list[list[int]],
                sp_stage_b: SamplingParams,
                *,
                max_greedy_steps: int = 500,
                tail_window_tokens: int = 24,
                debug: bool = True,
                debug_prefix: str = "[TwoStage]"
            ) -> list:
                # tokenizer = llm.get_tokenizer() #  NOTE SJ is this the error source?
                tokenizer = self.tokenizer
                results = []
                rank = int(dp_rank)
                print("rank=", rank)
                def dbg(msg: str):
                    if debug:
                        print(f"{debug_prefix}[rank={rank}] {msg}", flush=True)

                # Stage-A: strictly greedy, 1 token at a time
                # greedy_step_params = SamplingParams(
                #     temperature=0.0, top_p=1.0, top_k=-1,
                #     max_tokens=1, n=1, stop=None, detokenize=False, logprobs=None
                # )

                def sp_summary(sp: SamplingParams):
                    return (f"temp={getattr(sp, 'temperature', None)} "
                            f"top_p={getattr(sp, 'top_p', None)} "
                            f"top_k={getattr(sp, 'top_k', None)} "
                            f"n={getattr(sp, 'n', None)} "
                            f"max_tokens={getattr(sp, 'max_tokens', None)}")

                # len(prompt_token_ids_batch) = 4 questions per gpu (at most) 6*4+2*3=30 questions
                print('\ninit llm\n')
                llm = LLM(**model_args)
                for req_idx, prompt_ids in enumerate(prompt_token_ids_batch):
                    dbg(f"[req_idx={req_idx}] Stage-A begin | prompt_len={len(prompt_ids)}")
                    grown_prefix = []
                    matched_tag = None
                    matched_i = None

                    # ---- Stage A loop ----
                    for step in range(max_greedy_steps):
                        a_out = llm.generate(
                            prompt_token_ids=[prompt_ids + grown_prefix],
                            sampling_params=greedy_step_params,
                        )[0]
                        dbg(f"[req_idx={req_idx}] Stage-A call step={step} | request_id={a_out.request_id}")
                        # NOTE here

                        if not a_out.outputs or not a_out.outputs[0].token_ids:
                            dbg(f"[req_idx={req_idx}] Stage-A returned no token; breaking.")
                            break

                        new_id = a_out.outputs[0].token_ids[-1]
                        new_piece = tokenizer.decode([new_id], skip_special_tokens=False).replace("\n", "\\n")
                        grown_prefix.append(new_id)
                        dbg(f"[req_idx={req_idx}] greedy_token id={new_id} piece='{new_piece}' "
                            f"| prefix_len={len(grown_prefix)}")

                        found, tag, i_str, tail_text = _tail_has_think_tag(
                            tokenizer, prompt_ids + grown_prefix, max_tail_tokens=tail_window_tokens
                        )
                        if found:
                            matched_tag, matched_i = tag, i_str
                            preview = tail_text[-120:].replace("\n", "\\n")
                            dbg(
                                f"[req_idx={req_idx}] MATCH tag={matched_tag} (i={matched_i}) "
                                f"at step={step} | tail_preview='{preview}'"
                            )
                            #dbg(f"[req_idx={req_idx}] MATCH tag={matched_tag} (i={matched_i}) "
                            #    f"at step={step} | tail_preview='{tail_text[-120:].replace(chr(10),'\\n')}'")
                            break
                    # import rpdb
                    # port = 4444+rank
                    # print(f"Process {rank} waiting for debugger on port {port}")
                    # rpdb.set_trace(port=port)

                    # ---- STRICT: must have found a tag ----
                    # assert matched_tag is not None
                    # if matched_tag is None:
                    #     # Build a rich error with previews (helpful in RayTaskError)
                    #     prompt_preview = tokenizer.decode(
                    #         prompt_ids[-160:], skip_special_tokens=False
                    #     ).replace("\n", "\\n")
                    #     combined_tail = tokenizer.decode(
                    #         (prompt_ids + grown_prefix)[-200:], skip_special_tokens=False
                    #     ).replace("\n", "\\n")
                    #     recent_token_ids = grown_prefix[-16:]
                    #     msg = (
                    #         f"{debug_prefix}[rank={rank}] ERROR: No <thinki> found for req_idx={req_idx} "
                    #         f"after greedy probe (max_greedy_steps={max_greedy_steps}). "
                    #         f"prompt_len={len(prompt_ids)} add_prefix_len={len(grown_prefix)} "
                    #         f"| prompt_tail='{prompt_preview}' "
                    #         f"| combined_tail='{combined_tail}' "
                    #         f"| recent_prefix_token_ids={recent_token_ids}. "
                    #         f"Consider increasing tail_window_tokens if tokenizer splits the tag."
                    #     )
                    #     # Print once for logs, then raise to stop the run.
                    #     print(msg, flush=True)
                    #     raise RuntimeError(msg)

                    # ---- Stage B ----
                    sp = deepcopy(sp_stage_b)
                    dbg(f"[req_idx={req_idx}] Stage-B begin | add_prefix_len={len(grown_prefix)} "
                        f"| sampling=({sp_summary(sp)})")


                    # NOTE SJ: check force think2 performance
                    grown_prefix = [151644, 13708, 766, 17, 397]
                    ###########################################3
                    # import rpdb
                    # port = 4444+rank
                    # print(f"Process {rank} waiting for debugger on port {port}")
                    # rpdb.set_trace(port=port)

                    full_prompt_token_decoded = tokenizer.decode(prompt_ids+grown_prefix)
                    b_out = llm.generate(
                        prompt_token_ids=[prompt_ids + grown_prefix],
                        sampling_params=sp,
                    )[0]
                    dbg(f"[req_idx={req_idx}] Stage-B call | request_id={b_out.request_id}")

                    # import rpdb
                    # port = 4444+rank
                    # print(f"Process {rank} waiting for debugger on port {port}")
                    # rpdb.set_trace(port=port)

                    # Stitch prefix so downstream consumers see the full sequence
                    # NOTE prefix_text='<|im_start|><think2>\n'
                    prefix_text = tokenizer.decode(grown_prefix, skip_special_tokens=False) if grown_prefix else ""

                    for hyp_idx, g in enumerate(b_out.outputs):
                        g.text = prefix_text + g.text
                        if hasattr(g, "token_ids") and isinstance(g.token_ids, list):
                            g.token_ids = grown_prefix + g.token_ids
                        dbg(f"[req_idx={req_idx}] hypothesis={hyp_idx} final_len="
                            f"{len(g.token_ids) if hasattr(g,'token_ids') else 'NA'}")

                    b_out.prompt_token_ids = prompt_ids + grown_prefix
                    results.append(b_out)

                    dbg(f"[req_idx={req_idx}] DONE | matched_tag={matched_tag} "
                        f"| total_prompt_len={len(b_out.prompt_token_ids)}")

                return results

            return generate_until_think_then_sample(
                model_args=model_args,
                prompt_token_ids_batch=requests,
                sp_stage_b=sampling_params,
                max_greedy_steps=256,
                tail_window_tokens=24,
                debug=True,
                debug_prefix="[TwoStage]"
            )

        # -------- shard -> run -> gather (unchanged) --------
        requests_iter = (list(x) for x in distribute(self.data_parallel_size, inputs))
        call_args_iter = ((self.model_args, sampling_params, req, i) for i, req in enumerate(requests_iter))
        object_refs = [run_inference_one_model_remote.remote(*args) for args in call_args_iter]
        results_per_shard = ray.get(object_refs)

        ray.shutdown()

        outputs = [
            x
            for x in itertools.chain.from_iterable(
                itertools.zip_longest(*[list(chunk) for chunk in results_per_shard])
            )
            if x is not None
        ]

        return outputs


    def _generate(
        self,
        inputs: list[list[int]],
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[list[str]] = None,
        returns_logits: Optional[bool] = False,
        num_samples: int = 1,
        generate: bool = True,
        prompt_logprobs_k: int = 0,
    ) -> list[GenerativeResponse]:
        """Contains the actual logic of the generation."""
        sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())

        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            # sampling_params.logprobs = 1 if returns_logits else 0
            # sampling_params.prompt_logprobs = None
            if returns_logits is not None:
                sampling_params.logprobs = 0
            else:
                sampling_params.logprobs = None
            sampling_params.prompt_logprobs = prompt_logprobs_k # NOTE SJ: show prompt_logprobs with not ``None''
        else:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False

        #import ipdb; ipdb.set_trace()
        # NOTE len(inputs)=30
        # sampling_params.n=64
        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks
            # Hynek: With the newest vllm, it actually breaks when tensor_parallel_size == 1 and num_gpus not set,
            # as VLLM complains about no GPUs available.
            @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def run_inference_one_model(model_args: dict, sampling_params: SamplingParams, requests):
                print("model args", model_args)
                llm = LLM(**model_args)
                return llm.generate(prompt_token_ids=requests, sampling_params=sampling_params)

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, inputs)]
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            # NOTE len(list(inputs)) = 8
            # len(requests)=8, len(requests[0])=4, 
            # import ipdb; ipdb.set_trace()
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            outputs = [
                x
                for x in itertools.chain.from_iterable(itertools.zip_longest(*[list(x) for x in results]))
                if x is not None
            ]
        else:
            outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )

        return outputs

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )
        return self._loglikelihood_tokens(requests, override_bs=override_bs)

    def _loglikelihood_tokens(
        self,
        requests: list[LoglikelihoodRequest],
        override_bs: int = -1,
        return_bool_score: bool = True,
        rolling: bool = False,
    ) -> list[LoglikelihoodResponse]:
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=1)
        res = []

        for split in tqdm(dataset.splits_iterator()):
            # the last token is an eos token, so we don't need to add it
            inputs = [sample.tokenized_context + sample.tokenized_continuation for sample in split]
            # Left truncate the inputs to the maximum length
            inputs = [input[-self.max_length :] for input in inputs]
            outputs = self._generate(inputs, generate=False)

            for i, output in enumerate(outputs):
                input = split[i]
                continuation_logprobs = []
                for token, logprobs in zip(input.tokenized_continuation[::-1], output.prompt_logprobs[::-1]):
                    continuation_logprobs.append(logprobs[token])
                bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
                continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
                answer = LoglikelihoodResponse(
                    input_tokens=input.tokenized_context + input.tokenized_continuation,
                    generated_tokens=input.tokenized_continuation,
                    result=(sum(continuation_logprobs), bool_score if return_bool_score else None),
                )
                res.append(answer)

        return dataset.get_original_order(res)

    def loglikelihood_rolling():
        pass

    def loglikelihood_single_token():
        pass


class AsyncVLLMModel(VLLMModel):
    """VLLM models which deploy async natively (no ray). Supports DP and PP/TP but not batch size > 1"""

    DATASET_SPLITS = 1
    is_async = True

    def cleanup(self):
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def _create_auto_model(self, config: VLLMModelConfig) -> Optional[AsyncLLM]:
        """
        Creates an instance of the async vllm model loaded from HF. Requires using the v1 of VLLM.

        Returns:
            AsyncLLM: The created async VLLM instance
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "data_parallel_size": config.data_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "enforce_eager": True,
        }

        if config.data_parallel_size > 1:
            self._batch_size = "auto"

        model = AsyncLLM.from_engine_args(AsyncEngineArgs(**self.model_args))

        # If the max_length can't get extracted from the config, it will be inferred from the model
        if self._max_length is None:
            self._max_length = model.model_config.max_seq_len_to_capture

        return model

    async def _async_one_item(
        self,
        index: int,
        request: GreedyUntilRequest | LoglikelihoodRequest,
    ) -> Coroutine[None, list, str]:
        """Contains the actual logic of the generation."""
        sampling_params = SamplingParams(**self._config.generation_parameters.to_vllm_dict())

        if isinstance(request, LoglikelihoodRequest):
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False
            prompt = request.context + request.choice
            index = f"logprob_{index}"
        elif isinstance(request, GreedyUntilRequest):
            sampling_params.n = request.num_samples
            if sampling_params.n > 1:
                # Todo clementine: investigate more
                logger.warning(
                    "Careful, there can be unexpected behavior when using sampling evals with the async vllm model"
                )
            sampling_params.max_tokens = self._config.generation_parameters.max_new_tokens or request.generation_size
            sampling_params.stop = [] if self.use_chat_template else request.stop_sequence
            sampling_params.logprobs = int(request.use_logits)
            prompt = request.context
            index = f"generative_{index}"

        generator = self.model.generate(request_id=str(index), prompt=prompt, sampling_params=sampling_params)
        try:
            while output := await anext(generator):
                continue
        except StopAsyncIteration:
            pass

        return output

    async def _async_batch(self, requests: list[GreedyUntilRequest | LoglikelihoodRequest]) -> list:
        processed_requests = [
            self._async_one_item(index=index, request=request) for index, request in enumerate(requests)
        ]
        results = await asyncio.gather(*processed_requests)
        return results

    async def greedy_until(
        self,
        requests: list[GreedyUntilRequest],
        **kwargs,
    ) -> list[GenerativeResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.

        Returns:
            list[GenerateReturn]: list of generated responses.
        """
        for request in requests:
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]
            request.tokenized_context = self.tok_encode(request.context)

        results = []

        responses: list[RequestOutput] = await self._async_batch(requests=requests)

        for response in responses:
            output_token_ids = [outputs.token_ids for outputs in response.outputs]
            full_logprobs = [output.logprobs for output in response.outputs] or []
            logprobs = [logprob[token_id].logprob for token_id, logprob in zip(output_token_ids[0], full_logprobs[0])]
            result = [output.text for output in response.outputs]
            input_token_ids = response.prompt_token_ids

            cur_response = GenerativeResponse(
                result=result,
                logits=logprobs,
                generated_tokens=list(output_token_ids),
                input_tokens=input_token_ids,
            )
            results.append(cur_response)

        return results

    async def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        return_bool_score: bool = True,
        **kwargs,
    ) -> list[LoglikelihoodResponse]:
        """
        Generates responses using a greedy decoding strategy until certain ending conditions are met and
        stores the logprobs.

        Args:
            requests (list[Request]): list of requests containing the context and ending conditions.

        Returns:
            list[LoglikelihoodResponse]: list of generated responses.
        """

        for request in requests:
            if request.context == "":
                request.tokenized_context = [self.tokenizer.eos_token_id]
                request.tokenized_continuation = self.tok_encode(request.choice)
            else:
                # The following line is mandatory for compatibility with the harness
                request.tokenized_context, request.tokenized_continuation = self.tok_encode_pair(
                    request.context, request.choice, pairwise=self.pairwise_tokenization
                )

        results = []

        responses: list[RequestOutput] = await self._async_batch(requests=requests)

        for response, input in zip(responses, requests):
            continuation_logprobs = []
            for token, logprobs in zip(input.tokenized_continuation[::-1], response.prompt_logprobs[::-1]):
                continuation_logprobs.append(logprobs[token])
            bool_score = all(logprob.rank == 1 for logprob in continuation_logprobs)
            continuation_logprobs = [logprob.logprob for logprob in continuation_logprobs]
            answer = LoglikelihoodResponse(
                input_tokens=input.tokenized_context + input.tokenized_continuation,
                generated_tokens=input.tokenized_continuation,
                result=(sum(continuation_logprobs), bool_score if return_bool_score else None),
            )
            results.append(answer)

        return results

    def loglikelihood_rolling(self):
        pass

    def loglikelihood_single_token():
        pass
