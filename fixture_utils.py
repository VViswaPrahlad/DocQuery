"""Generate public evaluation PDFs from explicit synthetic text."""
import json
from pathlib import Path
import pymupdf

from config import ROOT

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "corpus.json"


def load_fixture(path=FIXTURE_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_fixture_pdfs(directory, fixture=None):
    fixture = fixture or load_fixture()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for document in fixture["documents"]:
        name = document["name"]
        if Path(name).name != name or not name.endswith(".pdf"):
            raise ValueError("Fixture PDF names must be plain PDF filenames")
        path = directory / name
        with pymupdf.open() as pdf:
            for text in document["pages"]:
                page = pdf.new_page()
                remaining = page.insert_textbox(pymupdf.Rect(72, 72, 520, 740), text, fontsize=12)
                if remaining < 0:
                    raise ValueError("Fixture text does not fit on its declared page")
            pdf.save(path)
        paths.append(str(path))
    return paths
