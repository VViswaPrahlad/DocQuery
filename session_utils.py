"""Short-lived, workspace-contained upload staging. PDFs are removed after ingestion."""
from contextlib import contextmanager
from pathlib import Path
import tempfile

from config import ROOT


@contextmanager
def stage_uploads(uploaded_files):
    runtime = ROOT / ".runtime" / "uploads"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="session_", dir=runtime) as directory:
        paths = []
        for uploaded in uploaded_files:
            # Handle either browser path separator; reject ambiguous collisions.
            name = uploaded.name.replace("\\", "/").rsplit("/", 1)[-1]
            if not name or name in {".", ".."} or not name.lower().endswith(".pdf"):
                raise ValueError("Uploads must have a valid PDF filename")
            path = Path(directory) / name
            if path.exists():
                raise ValueError(f"Duplicate upload filename: {name}")
            path.write_bytes(uploaded.getbuffer())
            paths.append(str(path))
        yield paths
