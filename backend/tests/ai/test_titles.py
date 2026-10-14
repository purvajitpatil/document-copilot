from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.ai import titles
from app.ai.titles import ThreadTitle, clean_title, suggest_thread_title


def test_clean_title_normalizes() -> None:
    assert clean_title('  "Apple Services Revenue Mix."\n') == "Apple Services Revenue Mix"
    assert clean_title("   ") is None
    assert clean_title("") is None


def test_clean_title_truncates_runaway_output() -> None:
    long = " ".join(f"w{i}" for i in range(60))
    assert len(clean_title(long).split()) <= 12


def test_suggest_title_returns_cleaned_model_title() -> None:
    with patch.object(titles, "structured_completion", return_value=ThreadTitle(title="NVDA Data Center Demand.")):
        assert asyncio.run(suggest_thread_title("How is NVIDIA data center demand?")) == (
            "NVDA Data Center Demand"
        )


def test_suggest_title_swallows_failures() -> None:
    with patch.object(titles, "structured_completion", side_effect=RuntimeError("down")):
        assert asyncio.run(suggest_thread_title("anything")) is None


def test_suggest_title_skips_blank_input() -> None:
    with patch.object(titles, "structured_completion") as completion:
        assert asyncio.run(suggest_thread_title("   ")) is None
    completion.assert_not_called()
