"""
title: Budget Forcing Reasoning Pipe
author: latent-variable
github: https://github.com/latent-variable/Simple-test-time-scaling
open-webui: https://openwebui.com/f/latentvariable/Simple-test-time-scaling/
Set up instructions: https://o1-at-home.hashnode.dev/run-o1-at-home-privately-think-respond-pipe-tutorial-with-open-webui-ollama
version: 0.1.0
description: s1: Simple test-time scaling pipeline for models like deepseek-r1 models with OpenAI/Ollama support.
Directly compatible with build in reasoning formater 
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
from open_webui.routers.ollama import generate_chat_completion as ollama_completion
from open_webui.routers.openai import generate_chat_completion as openai_completion
import logging

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
        # API Selection
        USE_OPENAI_REASONING: bool = Field(
            default=False,
            description="(NOT TESTED) Use OpenAI API instead of Ollama"
        )
        REASONING_MODEL: str = Field(
            default="deepseek-r1:8b",
            description="Model for reasoning phase (Ollama name or OpenAI ID)"
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

    async def get_response(self, model: str, messages: List[Dict], stream: bool):
        use_openai = self.valves.USE_OPENAI_REASONING
        try:
            if use_openai:
                response = await openai_completion(
                    self.__request__,
                    {"model": model, "messages": messages, "stream": stream},
                    user=self.__user__
                )
            else:
                response = await ollama_completion(
                    self.__request__,
                    {"model": model, "messages": messages, "stream": stream},
                    user=self.__user__
                )
            return response
        except Exception as e:
            logger.error(f"API Error ({'OpenAI' if use_openai else 'Ollama'}): {str(e)}")
            raise

    async def _handle_api_stream(self, response) -> AsyncGenerator[str, None]:
        buffer = ""
        async for chunk in response.body_iterator:
            if self.valves.USE_OPENAI_REASONING:
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        if 'choices' in data and data['choices']:
                            content = data['choices'][0]['delta'].get('content', '')
                            buffer += content
                            yield content
                    except json.JSONDecodeError:
                        continue
            else:
                buffer += chunk.decode()
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

    async def _generate_reasoning(self, messages: List[Dict], __event_emitter__) -> AsyncGenerator[str, None]:
        # original_messages remains constant; we update the conversation with our accumulated buffer.
        original_messages = messages.copy()
        self.buffer = ""
        self.generated_thinking_tokens = 0
        reached_maximum_thinking = False
        final_response_complete = False

        # Continue requesting responses until we get a final answer.
        try:
            while not final_response_complete:
                # Construct current conversation: original messages + our accumulated reasoning so far.
                current_messages = original_messages.copy()
                if self.buffer:
                    current_messages.append({"role": "assistant", "content": self.buffer})
                response = await self.get_response(
                    model=self.valves.REASONING_MODEL,
                    messages=current_messages,
                    stream=True
                )
                async for content in self._handle_api_stream(response):
                    status = f"{self.generated_thinking_tokens} reasoning tokens"
                    await self.emit_status(status, __event_emitter__, done=False)
                    # Just give final response 
                    if reached_maximum_thinking:
                         final_response_complete = True
                         yield content
                    else:
                        # If we exceed maximum allowed thinking tokens, force termination.
                        if self.generated_thinking_tokens >= self.valves.MAX_THINKING_TOKENS:
                            yield f"\n{self.valves.END_THINK_TOKEN}"
                            self.buffer += f"\n{self.valves.END_THINK_TOKEN}"
                            reached_maximum_thinking = True
                            break

                        if self.valves.END_THINK_TOKEN.strip() in content.strip():
                            # Model is trying to end its thinking.
                            if self.generated_thinking_tokens < self.valves.MIN_THINKING_TOKENS:
                                # Not enough tokens: replace end token with "Wait" and request more thinking.
                                content = content.replace(self.valves.END_THINK_TOKEN, "")
                                yield " Wait"
                                self.buffer += " Wait"
                                self.generated_thinking_tokens += 1
                                break  # Exit streaming to issue a new request with updated context.
                            else:
                                # Sufficient tokens have been generated; accept termination.
                                yield content
                                reached_maximum_thinking = True
                        else:
                            self.generated_thinking_tokens += 1
                            yield content
                            self.buffer += content

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