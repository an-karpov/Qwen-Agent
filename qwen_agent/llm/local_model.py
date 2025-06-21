# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import json
import time
import asyncio
from pprint import pformat
from typing import Dict, Iterator, List, Optional
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from starlette.responses import StreamingResponse


from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, USER, SYSTEM, Message
from qwen_agent.log import logger

MODELS_PATHS = {
    "llama-7b":'/gpfs/gpfs0/an.karpov/models/Meta-Llama-3-8B',
    "phi-3.5-mini":'/gpfs/gpfs0/an.karpov/models/Phi-3.5-mini-instruct',
    "deepseek-qwen-32b":"/gpfs/gpfs0/an.karpov/models/deepseek-qwen-32b",
    "Qwen/QwQ-32B":"Qwen/QwQ-32B",
    "Qwen/Qwen3-32B":"Qwen/Qwen3-32B",
    #"<your model>":"<local path to your model>"
}

MODELS = {}

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "phi-3.5-mini"
    messages: List[dict] = []
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

@register_llm('local')
class HFModel(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model_name = self.model or 'microsoft/Phi-3.5-mini-instruct'
        
        if self.model_name not in MODELS_PATHS:
            raise KeyError("Such model is not available")

        if self.model_name not in MODELS:
            model_path = MODELS_PATHS.get(self.model_name, self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left") 

            self.model = AutoModelForCausalLM.from_pretrained( 
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                # tp_plan="auto",
                trust_remote_code=True)
            MODELS[self.model_name] = (self.model, self.tokenizer)
        else:
            model, tokenizer = MODELS[self.model_name]
            self.model=model
            self.tokenizer=tokenizer

        self.pipe = pipeline( 
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16)

    def generate(self, messages, stream=False, **generate_cfg):
        messages_for_hf = self._format_messages_for_hf(messages)
        return self.get_response(messages_for_hf, stream=stream, generate_cfg=generate_cfg)
    
    def get_response(self, messages:list, stream:bool, generate_cfg:dict):
        num_attempts = 0
        result = None

        while True:

            if num_attempts==5:
                break

            try:
                result = self._get_response(messages, stream, generate_cfg)
                return result
            except ValueError as e:
                num_attempts += 1
                print(f"ERROR when trying process the result: {e}")
            finally:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return result

    def _get_response(self, messages:list, stream:bool, generate_cfg:dict):
        generation_args = { 
            "max_new_tokens": generate_cfg.get("max_tokens", 3000), 
            "return_full_text": False, 
            "temperature": generate_cfg.get("temperature", 0.6), 
            "do_sample": False, 
        }

        generate_cfg["model"] = self.model_name

        output = self.pipe(messages, **generation_args) 
        resp_content = output[0]['generated_text']
        # print(f"RAW CONTENT:\n{resp_content}\n------------\n")
        if stream:
            return _resp_generator(self._delete_think(resp_content), generate_cfg)
            # return StreamingResponse(
            #     _resp_async_generator(resp_content, prompt), media_type="application/x-ndjson"
            # )
    
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        created = int(time.time())
        
        return {
            "id": "1337",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"message": Message(role=ASSISTANT, content=self._delete_think(resp_content), reasoning_content=None)}],
        }

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        "TODO: need to fix"
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self.generate(messages=messages, stream=True, **generate_cfg)
            # print("Chat stream response: ", response)
            # if delta_stream:
            #     for chunk in response:
            #         if chunk.choices:
            #             if hasattr(chunk.choices[0].delta,
            #                        'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            #                 yield [
            #                     Message(role=ASSISTANT,
            #                             content='',
            #                             reasoning_content=chunk.choices[0].delta.reasoning_content)
            #                 ]
            #             if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            #                 yield [Message(role=ASSISTANT, content=chunk.choices[0].delta.content)]
            # else:
            full_response = ''
            full_reasoning_content = ''
            for chunk in response:
                print("Chunk: ", chunk)
                if len(chunk["choices"]):
                    if hasattr(chunk["choices"][0]["delta"],
                                'reasoning_content') and chunk["choices"][0]["delta"]["reasoning_content"]:
                        full_reasoning_content += chunk["choices"][0]["delta"]["reasoning_content"]
                    # if hasattr(chunk["choices"][0]["delta"], 'content') and chunk["choices"][0]["delta"]["content"]:
                    #     print("add to full response")
                    full_response += chunk["choices"][0]["delta"]["content"]
                    yield [Message(role=ASSISTANT, content=full_response, reasoning_content=full_reasoning_content)]
            logger.info(f'Full response: {full_response}')
        except ValueError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self.generate(messages=messages, stream=False, **generate_cfg)
            # return [response["choices"][0]["message"]]
            return [Message(role=ASSISTANT, content=response["choices"][0]["message"]["content"], reasoning_content=None)]
        except ValueError as ex:
            raise ModelServiceError(exception=ex)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [msg.model_dump() for msg in messages]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages

    def _delete_think(self, raw_result:str)->str:
        pattern = r"([\s|\S]*)</think>"
        return re.sub(pattern, "", raw_result).strip("\n")

    # def _format_messages_for_hf(self, messages: List[Message]) -> str:
    #     """Convert conversation messages to a single prompt string."""
    #     prompt = ""
    #     for msg in messages:
    #         if msg["role"] == SYSTEM:
    #             prompt += f"<|system|>\n{msg['content']}</s>\n"
    #         elif msg["role"] == USER:
    #             prompt += f"<|user|>\n{msg['content']}</s>\n"
    #         elif msg["role"] == ASSISTANT:
    #             prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    #     prompt += "<|assistant|>\n"
    #     return prompt

    def _format_messages_for_hf(self, messages: List[Message]) -> List[dict]:
        """Convert conversation messages to a single prompt string."""
        messages_for_hf = []
        for msg in messages:
            if msg["role"] == SYSTEM:
                messages_for_hf += [{"role": "system", "content": msg['content']}]
            elif msg["role"] == USER:
                messages_for_hf += [{"role": "user", "content": msg['content']}]
            elif msg["role"] == ASSISTANT:
                messages_for_hf += [{"role": "assistant", "content": msg['content']}]
        return messages_for_hf
    
    def _hf_stream_tokens(self):
        """Yield tokens from Hugging Face streamer if available."""
        if not hasattr(self, 'streamer'):
            raise ValueError("Streamer not initialized for delta streaming")
        
        for token in self.streamer:
            yield token

def _resp_generator(text_resp: str, generate_cfg:dict):
# let's pretend every word is a token and return it over time
    tokens = text_resp.split(" ")

    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": generate_cfg["model"],
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield chunk
        # await asyncio.sleep(1)
    # yield "data: [DONE]\n\n"

async def _resp_async_generator(text_resp: str, request:ChatCompletionRequest):
# let's pretend every word is a token and return it over time
    tokens = text_resp.split(" ")

    for i, token in enumerate(tokens):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": request.model,
            "choices": [{"delta": {"content": token + " "}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(1)
    yield "data: [DONE]\n\n"
