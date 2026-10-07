"""Workspace-local configuration; environment takes precedence over .env.

Streamlit secrets take precedence when present. Legacy GROK_* names remain
supported so existing local setups continue to work.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


def get_config(key, default=None, aliases=()):
    import streamlit as st
    names = (key, *aliases)
    try:
        for name in names:
            if name in st.secrets:
                return st.secrets[name]
    except (KeyError, FileNotFoundError):
        pass
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return default


def positive_int(key, default):
    try:
        value = int(get_config(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value
