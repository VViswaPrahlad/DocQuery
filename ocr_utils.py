"""
ocr_utils.py
------------
OCR fallback for scanned / image-based PDF pages.

Rasterizes pages using PyMuPDF's own page.get_pixmap() rather than
pdf2image, which avoids a separate poppler dependency -- poppler is
fiddly to install on Windows, whereas PyMuPDF is already a hard
dependency of ingest.py.

Tesseract itself is still a required SYSTEM binary (not a pip package)
-- see the OCR setup section in README.md.
"""

import io
import os
import platform
import warnings
import pymupdf as fitz
import pytesseract
from PIL import Image

# On Windows (local dev), pytesseract can't find tesseract.exe unless
# it's on PATH -- point at it explicitly. On Linux (Streamlit Community
# Cloud, installed via packages.txt), "tesseract" is already on PATH
# after apt-get install, so no override is needed there.
if platform.system() == "Windows":
    _default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default_win_path):
        pytesseract.pytesseract.tesseract_cmd = _default_win_path

# If a page's native text layer is shorter than this, treat the page as
# "no real text" and try OCR instead. A page with just a stray page
# number isn't truly empty but isn't real content either.
MIN_NATIVE_CHARS = 20

# Render zoom factor. 2.0 on a standard 72-dpi PDF page ~= 144 DPI,
# which is a reasonable accuracy/speed tradeoff for Tesseract. Bump to
# 3.0 for small/dense text at the cost of slower OCR.
OCR_ZOOM = 2.0


def check_tesseract_available() -> bool:
    """Call this once at app startup so a missing Tesseract install fails
    with a clear message instead of a cryptic error mid-ingestion."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def needs_ocr(native_text: str, min_chars: int = MIN_NATIVE_CHARS) -> bool:
    """Decide whether a page's native PyMuPDF text extraction is too
    thin to be real content, and OCR should be attempted instead."""
    return len(native_text.strip()) < min_chars


def ocr_page(page: fitz.Page, zoom: float = OCR_ZOOM, lang: str = "eng") -> str:
    """Rasterize a single PyMuPDF page to an image and run Tesseract on it."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(image, lang=lang)


def extract_page_text(page: fitz.Page, ocr_fallback: bool = True) -> tuple[str, str]:
    """
    Try native text extraction first (fast, exact). If the page comes back
    effectively empty, fall back to OCR. Returns (text, source) where
    source is "native" or "ocr" -- useful to surface in the UI so users
    can see which pages required OCR.
    """
    native_text = page.get_text("text")
    if not ocr_fallback or not needs_ocr(native_text):
        return native_text, "native"

    try:
        ocr_text = ocr_page(page)
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError, OSError) as exc:
        warnings.warn(
            f"OCR unavailable for page {page.number + 1} ({type(exc).__name__}); "
            "keeping native text. Image-only content on this page was not indexed.",
            RuntimeWarning, stacklevel=2,
        )
        return native_text, "native"
    # Only prefer OCR output if it actually found more than the native layer did --
    # guards against OCR producing garbage on a truly blank page.
    if len(ocr_text.strip()) > len(native_text.strip()):
        return ocr_text, "ocr"
    return native_text, "native"
