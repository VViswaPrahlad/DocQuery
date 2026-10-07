"""Validate citation references against retrieved metadata, not claim truth."""
import re

CITATION_PATTERN = re.compile(r"[\[【]\s*Source:\s*([^\]】\n]+?),\s*page\s+(\d+)\s*[\]】]", re.I)


def normalize_citations(answer):
    """Canonicalize model citation delimiters/spacing without changing evidence."""
    return CITATION_PATTERN.sub(lambda m: f"[Source: {m.group(1).strip()}, page {int(m.group(2))}]", answer or "")


def validate_citations(answer, sources):
    evidence = {}
    for source in sources:
        key = (source["doc_name"], int(source["page"]))
        evidence.setdefault(key, []).append(source.get("chunk_id"))
    valid, invalid = [], []
    references = CITATION_PATTERN.findall(answer or "")
    for name, page in dict.fromkeys((name.strip(), int(page)) for name, page in references):
        reference = {"doc_name": name, "page": page}
        if (name, page) in evidence:
            reference["chunk_ids"] = [c for c in evidence[(name, page)] if c]
            valid.append(reference)
        else:
            invalid.append(reference)
    status = "invalid" if invalid else "matched" if valid else "missing" if sources else "no_evidence"
    return {"status": status, "valid": valid, "invalid": invalid,
            "reference_precision": len(valid) / (len(valid) + len(invalid)) if references else None,
            "limitation": "Checks document/page membership in retrieved evidence; does not verify factual claims."}
