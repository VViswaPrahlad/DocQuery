"""Preview publication candidates and flag secrets without initializing Git.

This matcher supports this project's .gitignore rules; it is not a replacement
for Git's matcher. Rerun git status/check-ignore after any future initialization.
"""
import fnmatch
import json
import os
from pathlib import Path
import re

from config import ROOT


def ignore_reason(relative, rules, is_dir=False):
    path = relative.replace("\\", "/")
    reason = None
    for rule in rules:
        negate = rule.startswith("!")
        pattern = rule[1:] if negate else rule
        anchored = pattern.startswith("/")
        pattern = pattern.lstrip("/")
        directory = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        if directory:
            matched = path == pattern or path.startswith(pattern + "/")
            if "/" not in pattern and not anchored:
                matched = pattern in path.split("/")[:-1] or (is_dir and path.split("/")[-1] == pattern)
        elif "/" in pattern or anchored:
            matched = fnmatch.fnmatchcase(path, pattern)
        else:
            matched = any(fnmatch.fnmatchcase(part, pattern) for part in path.split("/"))
        if matched:
            reason = None if negate else rule
    return reason


def audit():
    rules = [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    candidates, ignored, findings = [], [], []
    known_secrets = []
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip("\"'")
                if re.search(r"KEY|TOKEN|SECRET|PASSWORD", key) and len(value) >= 12:
                    known_secrets.append(value)
    for directory, dirs, files in os.walk(ROOT):
        relative_dir = Path(directory).relative_to(ROOT)
        for name in list(dirs):
            rel = (relative_dir / name).as_posix()
            if name == ".git":
                dirs.remove(name)
                continue
            reason = ignore_reason(rel, rules, is_dir=True)
            # Do not prune a directory that has an explicit included descendant.
            has_include = any(rule.startswith("!" + rel + "/") for rule in rules)
            if reason and not has_include:
                ignored.append({"path": rel + "/", "rule": reason})
                dirs.remove(name)
        for name in files:
            rel = (relative_dir / name).as_posix()
            reason = ignore_reason(rel, rules)
            if reason:
                ignored.append({"path": rel, "rule": reason})
                continue
            candidates.append(rel)
            if Path(name).suffix.lower() in {".pdf", ".pkl", ".index", ".safetensors", ".bin"}:
                findings.append({"path": rel, "issue": "Unexpected document/model/index artifact eligible for publication"})
                continue
            content = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(content.splitlines(), 1):
                if any(secret in line for secret in known_secrets) or re.search(r"(?:gsk_|sk-|github_pat_|gh[pousr]_)[A-Za-z0-9_-]{20,}", line):
                    findings.append({"path": rel, "line": line_number, "issue": "Possible credential (value redacted)"})
    return {"candidates": sorted(candidates), "ignored": sorted(ignored, key=lambda r: r["path"]),
            "findings": findings, "git_initialized": (ROOT / ".git").exists(),
            "limitation": "Heuristic scan. Does not inspect ignored private documents; cannot guarantee absence of every secret."}


def main():
    report = audit()
    output = ROOT / ".runtime" / "privacy-audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
