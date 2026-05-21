"""
Unified LLM Client — Multi-Provider Support.

Supports: EFund, DeepSeek, OpenAI, and OpenAI-compatible custom providers.
Configure via .env or pass provider/model explicitly.

Usage:
    from src.llm_client import LLMClient

    # Auto-detect from LLM_PROVIDER env var (default: efund)
    client = LLMClient()
    answer = client.chat("What is Apple's revenue?", system="You are a financial analyst.")
    answer = client.extract_answer("What is Apple's revenue?", tool_outputs)

    # Explicit provider
    client = LLMClient(provider="deepseek")
    client = LLMClient(provider="openai", model="gpt-4o")

    # List available providers
    providers = LLMClient.list_providers()
"""

import os
import json
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
PROVIDER_CONFIGS = {
    "efund": {
        "base_url_env": "EFUNDS_BASE_URL",
        "api_key_env": "EFUNDS_API_KEY",
        "user_env": "EFUNDS_USER",
        "default_base_url": "https://aigc.efunds.com.cn/v1",
        "default_model": "EFundGPT-air",
        "source": "2025-SX",
        "label": "EFundGPT (易方达)",
    },
    "deepseek": {
        "base_url_env": "DEEPSEEK_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "label": "DeepSeek",
    },
    "openai": {
        "base_url_env": "OPENAI_BASE_URL",
        "api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "label": "OpenAI",
    },
    "custom": {
        "base_url_env": "CUSTOM_BASE_URL",
        "api_key_env": "CUSTOM_API_KEY",
        "model_env": "CUSTOM_MODEL",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "label": "Custom (OpenAI-compatible)",
    },
}


class LLMClient:
    """Unified LLM client supporting EFund, DeepSeek, OpenAI, and custom providers.

    Provider auto-detection:
    1. If `provider` arg is given, use it.
    2. Otherwise read LLM_PROVIDER from env (default: "efund").
    3. Falls back gracefully if no API key is set.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "efund").lower().strip()
        if self.provider not in PROVIDER_CONFIGS:
            raise ValueError(
                f"Unknown LLM provider: '{self.provider}'. "
                f"Available: {', '.join(PROVIDER_CONFIGS.keys())}"
            )

        self._cfg = PROVIDER_CONFIGS[self.provider]

        # Resolve base URL and API key from env
        self.base_url = os.getenv(self._cfg["base_url_env"], self._cfg["default_base_url"])
        self.api_key = os.getenv(self._cfg["api_key_env"], "")
        self.user = os.getenv(self._cfg.get("user_env", ""), "default_user")
        self.source = self._cfg.get("source", "")

        # Model: explicit arg > env var > provider default
        model_env = self._cfg.get("model_env", "")
        default_model = self._cfg["default_model"]
        self.model = model or os.getenv(model_env, default_model) if model_env else (model or default_model)

        # Build OpenAI client
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self.client is not None

    @property
    def label(self) -> str:
        return self._cfg.get("label", self.provider)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Single-turn chat. Returns response text or None on failure."""
        if not self.client:
            return None
        try:
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # EFund requires custom auth headers
            if self.provider == "efund":
                kwargs["extra_headers"] = {
                    "Efunds-User-Name": self.user,
                    "Efunds-Acc-Token": self.user,
                    "Efunds-Source": self.source,
                }
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM:{self.provider}] chat error: {e}")
            return None

    def chat_json(
        self,
        prompt: str,
        system: str = "You are a helpful assistant. Output only valid JSON.",
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Chat and parse response as JSON. Returns dict or None."""
        text = self.chat(prompt, system=system, temperature=temperature)
        if not text:
            return None
        try:
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_raw": text}

    def extract_answer(
        self,
        query: str,
        tool_outputs: List[str],
        max_output_len: int = 300,
    ) -> Tuple[Optional[str], float]:
        """Extract a financial answer from tool outputs.

        Returns (answer_text, confidence_0_to_1).
        Falls back to regex extraction if LLM is unavailable.
        """
        outputs_text = "\n".join([
            f"[{i}] {str(o)[:max_output_len]}"
            for i, o in enumerate(tool_outputs[-5:])  # last 5 outputs
        ])

        prompt = f"""Extract the final answer from these financial API outputs:

Query: {query}

Tool Outputs:
{outputs_text}

Return a JSON object with:
- "answer": the numeric or factual answer (be specific)
- "confidence": your confidence (0.0 to 1.0)
- "explanation": brief explanation (max 1 sentence)

Return ONLY valid JSON, no other text."""

        result = self.chat_json(prompt, system="You are a financial data analyst. Output only valid JSON.")
        if result and "answer" in result:
            conf = result.get("confidence", 0.5)
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 0.5
            return result["answer"], conf

        # Regex fallback
        import re
        numbers = re.findall(r'[\d,]+\.?\d*', outputs_text)
        ans = numbers[-1].replace(",", "") if numbers else "N/A"
        return ans, 0.5

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def list_providers() -> List[Dict[str, str]]:
        """Return available providers with their labels and models."""
        return [
            {
                "id": pid,
                "label": cfg["label"],
                "model": os.getenv(cfg.get("model_env", "")) or cfg["default_model"],
                "api_key_set": bool(os.getenv(cfg["api_key_env"], "")),
            }
            for pid, cfg in PROVIDER_CONFIGS.items()
        ]

    @staticmethod
    def detect_available() -> List[str]:
        """Return list of provider IDs that have API keys configured."""
        available = []
        for pid, cfg in PROVIDER_CONFIGS.items():
            if os.getenv(cfg["api_key_env"], ""):
                available.append(pid)
        return available

    @staticmethod
    def get_provider_configs():
        """Expose the PROVIDER_CONFIGS dict for external use."""
        return PROVIDER_CONFIGS


# ---------------------------------------------------------------------------
# Module-level convenience (uses LLM_PROVIDER env or efund default)
# ---------------------------------------------------------------------------
_default_client = None

def get_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Get or create a cached LLMClient. Pass provider to switch permanently."""
    global _default_client
    if provider:
        _default_client = LLMClient(provider=provider, model=model)
        return _default_client
    if model:
        # Model override without provider change — create new instance
        return LLMClient(provider=(_default_client.provider if _default_client else None), model=model)
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def chat(prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> Optional[str]:
    return get_client().chat(prompt, system=system, **kwargs)


def extract_answer(query: str, tool_outputs: List[str]) -> Tuple[Optional[str], float]:
    return get_client().extract_answer(query, tool_outputs)
