"""
Marswalk AI Assistant - LLM Client
Wrapper for Ollama API with tool calling support.
"""
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

import ollama
import httpx
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool that the LLM can call."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None


class LLMResponse(BaseModel):
    """Structured response from the LLM."""
    content: str
    tool_calls: List[ToolCall] = []
    model: str
    usage: Optional[Dict[str, int]] = None


class MarswalkLLM:
    """LLM client for Marswalk AI Assistant using Ollama."""

    def __init__(self, model: str = None, host: str = None):
        self.model = model or settings.DEFAULT_MODEL
        self.host = host or settings.OLLAMA_HOST
        self.client = ollama.AsyncClient(host=self.host)
        self._available_models: List[str] = []

    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = await self.client.list()
            self._available_models = [m.model for m in response.models]
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return self._available_models

    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            await self.client.pull(model)
            await self.list_models()
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    async def switch_model(self, model: str) -> bool:
        """Switch to a different model."""
        if model in await self.list_models():
            self.model = model
            return True
        return False

    def _build_tools_prompt(self, tools: List[ToolDefinition]) -> str:
        """Build system prompt with tool definitions for tool calling."""
        tools_json = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in tools
        ]

        return f"""You are the Marswalk AI Assistant, an intelligent mission control system for Mars exploration.
You have access to the following tools:

{json.dumps(tools_json, indent=2)}

When you need to use a tool, respond with a JSON object in this format:
{{
  "tool_calls": [
    {{
      "name": "tool_name",
      "arguments": {{"param": "value"}},
      "id": "call_123"
    }}
  ]
}}

Only use tools when necessary. If no tools are needed, respond normally with your analysis and answer.
All tool calls must be valid JSON. The results will be provided to you for continuation."""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolDefinition] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        stream: bool = False
    ) -> LLMResponse:
        """
        Send a chat request to Ollama with optional tool calling.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Available tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            LLMResponse with content and any tool calls
        """
        # Prepare messages for Ollama
        ollama_messages = []

        # Add system prompt with tools if provided
        if tools:
            system_prompt = self._build_tools_prompt(tools)
            ollama_messages.append({"role": "system", "content": system_prompt})

        ollama_messages.extend(messages)

        try:
            if stream:
                return await self._stream_chat(ollama_messages, temperature, max_tokens)
            else:
                response = await self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                )

                return self._parse_response(response)

        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise

    async def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Stream chat response from Ollama."""
        try:
            async for chunk in await self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=True
            ):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[Error: {str(e)}]"

    def _parse_response(self, response: Dict) -> LLMResponse:
        """Parse Ollama response into structured format."""
        message = response.get("message", {})
        content = message.get("content", "")

        # Check for tool calls in content
        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append(ToolCall(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", {}),
                    id=tc.get("id")
                ))
        else:
            # Try to parse tool calls from content (for models that don't use native tool calling)
            tool_calls = self._extract_tool_calls(content)
            if tool_calls:
                # Remove tool call JSON from content
                content = self._remove_tool_calls(content)

        return LLMResponse(
            content=content.strip(),
            tool_calls=tool_calls,
            model=response.get("model", self.model),
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
            }
        )

    def _extract_tool_calls(self, content: str) -> List[ToolCall]:
        """Extract tool calls from content (fallback for non-native tool calling)."""
        tool_calls = []
        try:
            # Strategy 1: Look for JSON with "tool_calls" key
            import re
            pattern = r'\{\s*"tool_calls"\s*:\s*\[.*?\]\s*\}'
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # Clean up JavaScript comments before parsing
                cleaned = self._strip_json_comments(match)
                try:
                    data = json.loads(cleaned)
                    for tc in data.get("tool_calls", []):
                        tool_calls.append(ToolCall(
                            name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                            id=tc.get("id")
                        ))
                except json.JSONDecodeError:
                    pass

            # Strategy 2: If no tool_calls found, try to parse entire content as JSON
            if not tool_calls:
                try:
                    cleaned = self._strip_json_comments(content.strip())
                    data = json.loads(cleaned)
                    if isinstance(data, dict) and "tool_calls" in data:
                        for tc in data.get("tool_calls", []):
                            tool_calls.append(ToolCall(
                                name=tc.get("name", ""),
                                arguments=tc.get("arguments", {}),
                                id=tc.get("id")
                            ))
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"Tool call extraction failed: {e}")
        return tool_calls

    def _strip_json_comments(self, text: str) -> str:
        """Remove JavaScript-style comments from JSON text."""
        import re
        # Remove // comments
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        # Remove /* */ comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text

    def _remove_tool_calls(self, content: str) -> str:
        """Remove tool call JSON from content."""
        import re
        pattern = r'\{\s*"tool_calls"\s*:\s*\[.*?\]\s*\}'
        return re.sub(pattern, '', content, flags=re.DOTALL).strip()

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama."""
        try:
            response = await self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response.get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return []


# Global LLM instance
llm_client = MarswalkLLM()