import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")

# Global Langfuse trace for the entire task
_langfuse_trace = None


def _get_or_create_langfuse_trace():
    """Get or create a Langfuse trace for the current task.
    
    This creates ONE trace per task (identified by LANGFUSE_SESSION_ID/TASK_ID),
    and all LLM calls become spans within that trace.
    """
    global _langfuse_trace
    
    if _langfuse_trace is not None:
        return _langfuse_trace
    
    try:
        from langfuse import Langfuse
        
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST")
        
        if not (public_key and secret_key):
            return None
        
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        
        # Use TASK_ID or SESSION_ID as trace identifier
        task_id = os.getenv("TASK_ID") or os.getenv("LANGFUSE_SESSION_ID") or "unknown"
        
        _langfuse_trace = langfuse.trace(
            name="geak-agent",
            session_id=task_id,
            metadata={
                "task_id": task_id,
                "model": os.getenv("MODEL_NAME", "unknown"),
            }
        )
        logger.info(f"Created Langfuse trace for task: {task_id}")
        return _langfuse_trace
        
    except ImportError:
        logger.debug("Langfuse not installed, skipping trace creation")
        return None
    except Exception as e:
        logger.warning(f"Failed to create Langfuse trace: {e}")
        return None


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""


class LitellmModel:
    def __init__(self, *, config_class: type = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self._langfuse_trace = None
        
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))
        
        # Initialize Langfuse trace if configured
        self._setup_langfuse()
    
    def _setup_langfuse(self):
        """Set up Langfuse tracing for LLM calls."""
        callbacks_env = os.getenv("LITELLM_CALLBACKS", "")
        if "langfuse" not in callbacks_env.lower():
            return
        
        # Get or create the global trace
        self._langfuse_trace = _get_or_create_langfuse_trace()
        if self._langfuse_trace:
            logger.info("Langfuse tracing enabled - all LLM calls will be spans in the task trace")

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        
        # Create a generation span if Langfuse trace exists
        generation = None
        if self._langfuse_trace:
            try:
                generation = self._langfuse_trace.generation(
                    name=f"llm-call-{self.n_calls + 1}",
                    model=self.config.model_name,
                    input=messages,
                )
            except Exception as e:
                logger.debug(f"Failed to create Langfuse generation: {e}")
        
        response = self._query(messages, **kwargs)
        
        # Calculate cost
        try:
            cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
            logger.warning(
                f"Error calculating cost for model {self.config.model_name}: {e}. "
                "Cost will be set to 0. For accurate cost tracking, check the model registry at "
                "https://klieret.short.gy/litellm-model-registry"
            )
            cost = 0.0
        
        self.n_calls += 1
        if cost < 0:
            logger.warning(f"Cost is negative: {cost}, setting to 0")
            cost = 0.0
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        content = response.choices[0].message.content or ""
        
        # End the generation span with output
        if generation:
            try:
                usage = response.usage
                generation.end(
                    output=content,
                    usage={
                        "input": usage.prompt_tokens if usage else 0,
                        "output": usage.completion_tokens if usage else 0,
                    },
                    metadata={"cost": cost},
                )
            except Exception as e:
                logger.debug(f"Failed to end Langfuse generation: {e}")
        
        return {
            "content": content,
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
