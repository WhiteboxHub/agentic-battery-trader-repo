"""
LLM configuration loader for the battery performance multi-agent pipeline.

Priority order (highest wins):
  1. Environment variables (LLM_PROVIDER, OLLAMA_MODEL, etc.)
  2. config.yaml in the project root
  3. Built-in defaults (openrouter / gpt-4o)

Usage:
    from agent.config import get_llm_config
    cfg = get_llm_config("analyst")   # returns LLMConfig for the Analyst Agent
    cfg = get_llm_config()            # default config (no per-agent override)

Switching to Ollama — two ways:

  Option A: Edit config.yaml
    provider: ollama

  Option B: Environment variable (overrides config.yaml)
    LLM_PROVIDER=ollama python main.py data/...

  Option C: Full override via env vars
    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://localhost:11434/v1
    OLLAMA_MODEL=gemma4:e4b-it-q4_K_M
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# config.yaml is always at the project root (one level above agent/)
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# LLMConfig dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Complete LLM connection configuration for one agent."""

    provider: str          # "openrouter" | "openai" | "ollama"
    base_url: str          # API base URL
    model: str             # Model identifier
    api_key: str           # API key (dummy "ollama" for local)
    json_mode_supported: bool = True   # Whether response_format=json_object works
    tool_calling_supported: bool = True  # Whether function calling works

    def __str__(self) -> str:
        key_hint = self.api_key[:12] + "..." if len(self.api_key) > 12 else self.api_key
        return f"{self.provider} / {self.model} @ {self.base_url} (key: {key_hint})"


# ─────────────────────────────────────────────────────────────────────────────
# YAML loader (stdlib only — no pyyaml dependency required for simple cases)
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader — handles the simple key: value structure in config.yaml.

    Falls back gracefully if PyYAML is installed (preferred) or parses manually.
    Only supports the subset of YAML used in config.yaml (no lists at top level,
    no anchors, no complex types).
    """
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    # Fallback: manual parser for simple key: value and nested sections
    result: dict = {}
    current_section: Optional[str] = None
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            # Skip comments and blank lines
            if not line or line.lstrip().startswith("#"):
                continue
            # Detect indented key (section entry)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if ":" not in stripped:
                continue
            key_part, _, val_part = stripped.partition(":")
            key = key_part.strip()
            val = val_part.strip()
            # Skip inline comments
            if "#" in val:
                val = val[:val.index("#")].strip()
            if indent == 0:
                if val == "" or val is None:
                    # Section header
                    current_section = key
                    if key not in result:
                        result[key] = {}
                else:
                    result[key] = _cast(val)
                    current_section = None
            elif current_section is not None:
                if isinstance(result.get(current_section), dict):
                    result[current_section][key] = _cast(val)
    return result


def _cast(value: str):
    """Convert string YAML scalar to Python bool/None/int/float/str."""
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "~", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Strip surrounding quotes
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Main config resolver
# ─────────────────────────────────────────────────────────────────────────────

_yaml_cache: dict | None = None


def _get_yaml() -> dict:
    global _yaml_cache
    if _yaml_cache is None:
        _yaml_cache = _load_yaml(_CONFIG_PATH)
    return _yaml_cache


def get_llm_config(agent_name: str = "") -> LLMConfig:
    """Return LLMConfig for the given agent, applying all override layers.

    Parameters
    ----------
    agent_name : One of "analyst", "market", "writer", "critic".
                 Pass empty string (default) for the pipeline default.

    Returns
    -------
    LLMConfig ready for use in _make_openai_client().
    """
    yaml_cfg = _get_yaml()

    # ── 1. Determine provider ────────────────────────────────────────────────
    provider = (
        os.environ.get("LLM_PROVIDER")
        or str(yaml_cfg.get("provider", "openrouter"))
    ).lower()

    # ── 2. Load provider-level defaults from config.yaml ────────────────────
    provider_cfg: dict = yaml_cfg.get(provider, {}) or {}

    # ── 3. Determine model (per-agent override → provider default → env var) ─
    agents_cfg: dict = yaml_cfg.get("agents", {}) or {}
    per_agent_model: str = agents_cfg.get(agent_name, "") if agent_name else ""

    if provider == "ollama":
        default_model = str(provider_cfg.get("model", "gemma4:e4b-it-q4_K_M"))
        model = (
            os.environ.get("OLLAMA_MODEL")
            or per_agent_model
            or default_model
        )
        base_url = (
            os.environ.get("OLLAMA_BASE_URL")
            or str(provider_cfg.get("base_url", "http://localhost:11434/v1"))
        )
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")  # dummy key
        json_mode = bool(provider_cfg.get("json_mode_supported", False))
        tool_calling = bool(provider_cfg.get("tool_calling_supported", True))

    elif provider == "openai":
        default_model = str(provider_cfg.get("model", "gpt-4o"))
        model = (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("MODEL")
            or per_agent_model
            or default_model
        )
        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or str(provider_cfg.get("base_url", "https://api.openai.com/v1"))
        )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        json_mode = True
        tool_calling = True

    else:  # openrouter (default)
        default_model = str(provider_cfg.get("model", "openai/gpt-4o"))
        model = (
            os.environ.get("MODEL")
            or per_agent_model
            or default_model
        )
        base_url = (
            os.environ.get("OPENROUTER_BASE_URL")
            or str(provider_cfg.get("base_url", "https://openrouter.ai/api/v1"))
        )
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in your .env file, "
                "or switch to provider: ollama in config.yaml."
            )
        json_mode = True
        tool_calling = True

    return LLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        json_mode_supported=json_mode,
        tool_calling_supported=tool_calling,
    )
