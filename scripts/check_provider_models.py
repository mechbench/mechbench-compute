#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

from mechbench_compute.providers.table_check import ServedModel, compare

KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def get(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers={"accept": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.load(res)


def anthropic(key: str) -> list[ServedModel]:
    out, after = [], None
    while True:
        page = get("https://api.anthropic.com/v1/models?limit=1000" + (f"&after_id={after}" if after else ""),
                   {"x-api-key": key, "anthropic-version": "2023-06-01"})
        out += [ServedModel(m["id"]) for m in page["data"]]
        if not page.get("has_more"):
            return out
        after = page["last_id"]


def openai(key: str) -> list[ServedModel]:
    page = get("https://api.openai.com/v1/models", {"authorization": f"Bearer {key}"})
    return [ServedModel(m["id"], shutdown=m.get("shutdown_date")) for m in page["data"]]


def gemini(key: str) -> list[ServedModel]:
    out, token = [], ""
    while True:
        page = get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
                   + (f"&pageToken={token}" if token else ""), {"x-goog-api-key": key})
        out += [ServedModel(m["name"].removeprefix("models/")) for m in page.get("models", [])]
        token = page.get("nextPageToken", "")
        if not token:
            return out


def xai(key: str) -> list[ServedModel]:
    page = get("https://api.x.ai/v1/language-models", {"authorization": f"Bearer {key}"})

    def per_million(cents_per_100m: int | None) -> float | None:
        return None if not cents_per_100m else cents_per_100m / 10_000

    out = []
    for m in page["models"]:
        rates = {k: v for k, v in {
            "input": per_million(m.get("prompt_text_token_price")),
            "cache_read": per_million(m.get("cached_prompt_text_token_price")),
            "output": per_million(m.get("completion_text_token_price")),
        }.items() if v is not None}
        long_rates = {k: v for k, v in {
            "input": per_million(m.get("prompt_text_token_price_long_context")),
            "cache_read": per_million(m.get("cached_prompt_text_token_price_long_context")),
            "output": per_million(m.get("completion_text_token_price_long_context")),
        }.items() if v is not None}
        above = m.get("long_context_threshold") or None
        out.append(ServedModel(m["id"], tuple(m.get("aliases") or ()), rates=rates,
                               long_context_above=None if above is None else above - 1,
                               long_rates=long_rates or None))
    return out


def deepseek(key: str) -> list[ServedModel]:
    page = get("https://api.deepseek.com/models", {"authorization": f"Bearer {key}"})
    return [ServedModel(m["id"]) for m in page["data"]]


LISTERS = {"anthropic": anthropic, "openai": openai, "gemini": gemini, "xai": xai,
           "deepseek": deepseek}


def main() -> int:
    problems = 0
    for provider, env in KEYS.items():
        key = os.environ.get(env)
        if not key:
            print(f"{provider}: skipped, {env} is not set")
            continue
        try:
            found = compare(provider, LISTERS[provider](key))
        except Exception as e:
            print(f"{provider}: could not list models: {e}")
            problems += 1
            continue
        if not found.problems:
            print(f"{provider}: the table agrees with the provider's models API")
        for p in found.problems:
            print(f"{provider}: {p}")
        problems += len(found.problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
