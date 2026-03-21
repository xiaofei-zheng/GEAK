"""
OpenAI Compatible Model - 支持自定义 endpoint 的模型类
用于连接 LiteLLM Gateway 或任何 OpenAI 兼容的 API
"""
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("openai_compatible")


@dataclass
class OpenAICompatibleConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    api_base: str | None = None
    cost_per_1k_input_tokens: float = 0.01
    cost_per_1k_output_tokens: float = 0.01
    # 兼容 get_model 自动添加的参数（Anthropic 模型会自动设置）
    set_cache_control: str | None = None


class OpenAICompatibleModel:
    """
    使用标准 OpenAI 客户端连接任何 OpenAI 兼容的 API endpoint。
    
    配置示例:
    ```yaml
    model:
      model_class: openai_compatible
      model_name: gpt-5.2
      model_kwargs:
        api_base: "http://litellm-service:4000/v1"
        api_key: "sk-xxx"
        temperature: 0.0
        max_tokens: 16000
    ```
    """
    
    def __init__(self, **kwargs):
        # Extract api_key and api_base from model_kwargs if present
        model_kwargs = kwargs.get("model_kwargs", {})
        
        if "api_key" in model_kwargs and model_kwargs["api_key"] is not None:
            if "api_key" not in kwargs or kwargs.get("api_key") is None:
                kwargs["api_key"] = model_kwargs.pop("api_key")
            else:
                model_kwargs.pop("api_key", None)
                
        if "api_base" in model_kwargs and model_kwargs["api_base"] is not None:
            if "api_base" not in kwargs or kwargs.get("api_base") is None:
                kwargs["api_base"] = model_kwargs.pop("api_base")
            else:
                model_kwargs.pop("api_base", None)

        self.config = OpenAICompatibleConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "dummy"
        api_base = self.config.api_base or os.getenv("OPENAI_API_BASE")
        
        if not api_base:
            raise ValueError("api_base is required for OpenAICompatibleModel. Set it in model_kwargs or OPENAI_API_BASE env var.")

        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        
        logger.info(f"Initialized OpenAICompatibleModel with base_url={api_base}, model={self.config.model_name}")

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, openai.AuthenticationError, openai.NotFoundError)),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Supported parameters for chat completions
        supported_params = {
            "temperature", "max_tokens", "top_p", "frequency_penalty",
            "presence_penalty", "stop", "stream", "n", "seed",
            "response_format", "tools", "tool_choice", "logprobs",
            "top_logprobs", "user"
        }

        # Merge config kwargs with call kwargs, filter unsupported params
        all_kwargs = {**self.config.model_kwargs, **kwargs}
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items()
            if k in supported_params
        }

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **filtered_kwargs
        )

        return response

    def _parse_response(self, response):
        content = ""
        try:
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Failed to parse response content: {e}")
        return content

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        content = self._parse_response(response)

        # Calculate cost
        cost = 0.0
        try:
            if response.usage:
                cost = (
                    (response.usage.prompt_tokens / 1000) * self.config.cost_per_1k_input_tokens +
                    (response.usage.completion_tokens / 1000) * self.config.cost_per_1k_output_tokens
                )
        except Exception as e:
            logger.warning(f"Failed to calculate cost: {e}")

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": content,
            "extra": {"response": response.model_dump()},
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


if __name__ == "__main__":
    # Test the model
    model = OpenAICompatibleModel(
        model_name="gpt-4.1",  # 使用可用的模型
        model_kwargs={
            "api_base": "http://litellm-service.primus-safe.svc.cluster.local:4000/v1",
            "api_key": "sk-owqZAsPxvY-RrjmHIJL5eQ",
            "temperature": 0.0,
            "max_tokens": 100,
        }
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    
    response = model.query(messages)
    print(f"Response: {response['content']}")
    print(f"Cost: ${model.cost:.4f}")
