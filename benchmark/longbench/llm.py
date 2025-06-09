import requests
import logging
import time
from pprint import pformat
from typing import Dict, Iterator, List, Optional
from http import HTTPStatus

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.log import logger


@register_llm('hf_model_server')
class HF_model_server(BaseFnCallModel):

    def __init__(self, url, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.url = url

    def generate(self, prompt):

        MAX_TRY = 3
        count = 0
        while count < MAX_TRY:
            response = requests.post(
                 self.url,
                 params={"prompt": prompt}
                 )
            if response.status_code == 200:
                output = response.json()
                return output
            else:
                err = 'Error code: %s, error message: %s' % (
                    response.status_code,
                    response.test,
                )
                logging.error(err)
                count += 1
            time.sleep(1)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)

        response = self.generate(messages=messages)
        if delta_stream:
            for chunk in response:
                if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    yield [Message(ASSISTANT, chunk.choices[0].delta.content)]
        else:
            full_response = ''
            for chunk in response:
                if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    yield [Message(ASSISTANT, full_response)]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)

        response = self.generate(messages=messages)
        return [Message(ASSISTANT, response.choices[0].message.content)]

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        messages = [msg.model_dump() for msg in messages]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages
