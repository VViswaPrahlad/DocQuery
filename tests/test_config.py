import streamlit as st
from config import get_config, positive_int
import pytest


def test_canonical_and_legacy_provider_names(monkeypatch):
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("TEST_LEGACY_KEY", "legacy-test-value")
    monkeypatch.delenv("TEST_CANONICAL_KEY", raising=False)
    assert get_config("TEST_CANONICAL_KEY", aliases=("TEST_LEGACY_KEY",)) == "legacy-test-value"
    monkeypatch.setenv("TEST_CANONICAL_KEY", "canonical-test-value")
    assert get_config("TEST_CANONICAL_KEY", aliases=("TEST_LEGACY_KEY",)) == "canonical-test-value"


def test_secrets_precede_process_environment(monkeypatch):
    monkeypatch.setattr(st, "secrets", {"TEST_SETTING": "secret-setting"})
    monkeypatch.setenv("TEST_SETTING", "environment-setting")
    assert get_config("TEST_SETTING") == "secret-setting"


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_positive_configuration_is_validated(monkeypatch, value):
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("TEST_INTEGER", value)
    with pytest.raises(ValueError, match="positive integer"):
        positive_int("TEST_INTEGER", 5)
