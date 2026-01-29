#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import crawl


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INFO_JSON = REPO_ROOT / "Crawl" / "json" / "info.json"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}") from e


def _bool_flag(name: str, enabled: bool) -> List[str]:
    return [name] if enabled else []


def _build_argv(cfg: Dict[str, Any]) -> List[str]:
    argv: List[str] = []

    # Inputs
    if cfg.get("input1"):
        argv += ["--input1", str(cfg["input1"])]
    if cfg.get("input2"):
        argv += ["--input2", str(cfg["input2"])]

    # Output
    if cfg.get("output"):
        argv += ["--output", str(cfg["output"])]

    # Common runtime flags
    argv += _bool_flag("--headless", bool(cfg.get("headless", False)))
    argv += _bool_flag("--resume", bool(cfg.get("resume", False)))
    argv += _bool_flag("--interactive-setup", bool(cfg.get("interactive_setup", False)))
    argv += _bool_flag("--allow-fallback-below-threshold", bool(cfg.get("allow_fallback_below_threshold", False)))

    # Numeric/param flags (only if present)
    if "min_match_score" in cfg:
        argv += ["--min-match-score", str(cfg["min_match_score"])]
    if "max_matches_per_sku" in cfg:
        argv += ["--max-matches-per-sku", str(cfg["max_matches_per_sku"])]

    # Optional: location/radius
    if "city" in cfg:
        argv += ["--city", str(cfg["city"])]
    if "province" in cfg:
        argv += ["--province", str(cfg["province"])]
    if "country" in cfg:
        argv += ["--country", str(cfg["country"])]
    if "radius_km" in cfg:
        argv += ["--radius-km", str(cfg["radius_km"])]

    # Optional: delays / scrolling / verify
    if "profile_dir" in cfg:
        argv += ["--profile-dir", str(cfg["profile_dir"])]
    if "max_scrolls" in cfg:
        argv += ["--max-scrolls", str(cfg["max_scrolls"])]
    if "verify_top_k" in cfg:
        argv += ["--verify-top-k", str(cfg["verify_top_k"])]
    if "min_delay" in cfg:
        argv += ["--min-delay", str(cfg["min_delay"])]
    if "max_delay" in cfg:
        argv += ["--max-delay", str(cfg["max_delay"])]

    return argv


def main() -> None:
    # Allow: python3 main.py [optional_config_path.json] [any extra CLI overrides...]
    cfg_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) >= 2 and sys.argv[1].endswith(".json") else DEFAULT_INFO_JSON
    cfg = _load_config(cfg_path)

    argv_from_json = _build_argv(cfg)

    # Allow overriding/adding flags at runtime:
    # python3 main.py json/info.json --resume --headless
    extra_cli = []
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        extra_cli = sys.argv[2:]
    else:
        extra_cli = sys.argv[1:]

    final_argv = argv_from_json + extra_cli

    args = crawl.parse_args(final_argv)
    crawl.run(args)


if __name__ == "__main__":
    main()
