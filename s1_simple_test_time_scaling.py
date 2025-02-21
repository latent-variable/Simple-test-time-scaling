"""
title: Budget Forcing Reasoning Pipe (Direct Ollama API)
author: latent-variable
github: https://github.com/latent-variable/Simple-test-time-scaling
open-webui: https://openwebui.com/f/latentvariable/budget_forcing_reasoning_pipe
Set up instructions: https://o1-at-home.hashnode.dev/run-o1-at-home-privately-think-respond-pipe-tutorial-with-open-webui-ollama
version: 0.2.0
description: s1: Simple test-time scaling pipeline for models like deepseek-r1 models using a direct Ollama API call.
Directly compatible with build in reasoning formatter.
Compatible: open-webui v0.5.x
# Acknowledgments
https://arxiv.org/pdf/2501.19393 s1: Simple test-time scaling paper
"""

import os
import json
from typing import Dict, List, AsyncGenerator
import asyncio
from pydantic import BaseModel, Field
from dataclasses import dataclass
from fastapi import Request
import aiohttp
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

@dataclass
class User:
    id: str
    email: str
    name: str
    role: str

class Pipe:
    class Valves(BaseModel):
        REASONING_MODEL: str = Field(
            default="deepseek-r1:8b",
            description="Model for reasoning phase (Ollama model id)"
        )
        OLLAMA_API_URL: str = Field(
            default="http://host.docker.internal:11434/api/chat",
            description="Direct API URL for Ollama (e.g. http://host.docker.internal:11434/api/chat)"
        )
        CONTEXT_SIZE: int = Field(
            default=2048,
            description="Context size to pass to the Ollama API (used as options.num_ctx)"
        )
        # Budget forcing parameters (from the s1 paper)
        MIN_THINKING_TOKENS: int = Field(
            default=100,
            description="Minimum number of thinking tokens required before termination is allowed"
        )
        MAX_THINKING_TOKENS: int = Field(
            default=300,
            description="Maximum number of thinking tokens allowed before forcing termination"
        )
        START_THINK_TOKEN: str = Field(
            default="<think>",
            description="Token indicating reasoning phase start"
        )
        END_THINK_TOKEN: str = Field(
            default="</think>",
            description="Token indicating reasoning phase end"
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.__user__ = None
        self.__request__ = None
        self._reset_state()

    def _reset_state(self):
        self.generated_thinking_tokens = 0
        self.buffer = ""

    def pipes(self):
        name = f"BudgetForcing_{self.valves.REASONING_MODEL}"
        return [{"name": name, "id": name}]

    @asynccontextmanager
    async def get_response(self, model: str, messages: List[Dict], stream: bool):
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": self.valves.CONTEXT_SIZE
            }
        }
        api_url = self.valves.OLLAMA_API_URL
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                response.raise_for_status()
                yield response

    async def _handle_api_stream(self, response) -> AsyncGenerator[str, None]:
        """
        Process the response stream directly from Ollama.
        Assumes the response is NDJSON, one JSON object per line.
        """
        buffer = ""
        async for chunk in response.content.iter_chunked(1024):
            text = chunk.decode('utf-8')
            buffer += text
            if "\n" in buffer:
                lines = buffer.split("\n")
                for line in lines[:-1]:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            yield data.get("message", {}).get("content", "")
                        except json.JSONDecodeError:
                            continue
                buffer = lines[-1]
        await response.release()

    async def _generate_reasoning(self, messages: List[Dict], __event_emitter__) -> AsyncGenerator[str, None]:
        original_messages = messages.copy()
        self.buffer = ""
        self.generated_thinking_tokens = 0
        reached_maximum_thinking = False
        final_response_complete = False

        try:
            while not final_response_complete:
                current_messages = original_messages.copy()
                if self.buffer:
                    current_messages.append({"role": "assistant", "content": self.buffer})
                async with self.get_response(self.valves.REASONING_MODEL, current_messages, stream=True) as response:
                    async for content in self._handle_api_stream(response):
                        status = f"{self.generated_thinking_tokens} reasoning tokens"
                        await self.emit_status(status, __event_emitter__, done=False)
                        if self.generated_thinking_tokens >= self.valves.MAX_THINKING_TOKENS:
                            yield f"\n{self.valves.END_THINK_TOKEN}"
                            self.buffer += f"\n{self.valves.END_THINK_TOKEN}"
                            reached_maximum_thinking = True
                            break

                        if self.valves.END_THINK_TOKEN.strip() in content.strip():
                            if self.generated_thinking_tokens < self.valves.MIN_THINKING_TOKENS:
                                content = content.replace(self.valves.END_THINK_TOKEN, "")
                                print("Extending")
                                yield " Wait"
                                self.buffer += " Wait"
                                self.generated_thinking_tokens += 1
                                break
                            else:
                                yield content
                                reached_maximum_thinking = True
                                break
                        else:
                            self.generated_thinking_tokens += 1
                            yield content
                            self.buffer += content
                if reached_maximum_thinking:
                    final_response_complete = True
        except Exception as e:
            status = f"Reasoning error: {str(e)}"
            await self.emit_status(status, __event_emitter__, done=False)

    async def pipe(self, body: dict, __user__: dict, __event_emitter__, __request__: Request, __task__=None) -> AsyncGenerator[str, None]:
        self.__user__ = User(**__user__)
        self.__request__ = __request__
        self._reset_state()

        if __task__ is not None:
            yield body["messages"][:20]
        else:
            try:
                async for content in self._generate_reasoning(body["messages"], __event_emitter__):
                    yield content
                status = f"Completed with {self.generated_thinking_tokens} reasoning tokens"
                await self.emit_status(status, __event_emitter__, done=True)
            except Exception as e:
                status = f"Pipeline error: {str(e)}"
                await self.emit_status(status, __event_emitter__, done=True)
            yield ""

    async def emit_status(self, status, __event_emitter__, done=False):
        await __event_emitter__({
            "type": "status",
            "data": {"description": status, "done": done}
        })