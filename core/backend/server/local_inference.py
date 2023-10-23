from typing import Any, List, Mapping, Optional

import requests
from langchain.llms.base import LLM
from langchain.schema import ChatMessage
from loguru import logger

__all__ = ["LocalLlama2GPTQ"]

SystemMessage = lambda content: ChatMessage(content=content, role="system")
HumanMessage = lambda content: ChatMessage(content=content, role="user")


class ChatBot:
    endpoint = "/v1/chat/completions"

    def __init__(self, url: str):
        self.api = f"{url}{self.endpoint}"

    def chat(self, message: str) -> str:
        system = SystemMessage("You are a helpful assistant.")  # TODO: customize
        human = HumanMessage(message)
        messages = system + human

        response = requests.post(self.api, messages.json())

        try:
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(e)


class LocalLlama2GPTQ(LLM):
    url: str = None
    chatbot: ChatBot = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        if self.chatbot is None:
            self.chatbot = ChatBot(self.url)

        return self.chatbot.chat(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": "Llama-2-13B-chat-GPTQ"}
