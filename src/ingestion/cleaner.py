"""
cleaner.py -- Generic text cleaning and multi-signal structural section parsing.

No document-specific logic. Works on any raw text regardless of domain.
Supports: Markdown (#, ##, ###), major headings, numbered/bulleted sub-items,
typography signals, and hierarchy preservation.
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text for downstream chunking.

    Operations performed (all generic, domain-independent):
    1. Normalize line endings to \n
    2. Normalize horizontal whitespace (spaces/tabs)
    3. Collapse excessive blank lines (max 2 consecutive)
    4. Strip leading/trailing whitespace
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document_title(text: str, filename: str = "") -> str:
    """
    Extract canonical document title from text structure, with clean filename fallback.
    Completely domain-agnostic.

    Priority:
    1. Explicit document title in content (Markdown H1 or document header label in first 5 lines)
    2. Standalone short first line (title style)
    3. Cleaned filename stem (e.g. 'annual_report.txt' -> 'Annual Report')

    Also detects if an acronym/abbreviation in parentheses follows the title in the opening text
    (e.g., 'United Nations Development Programme (UNDP)').

    Args:
        text:     Cleaned raw text of the document.
        filename: Original filename (used for fallback).

    Returns:
        Canonical document title string.
    """
    from pathlib import Path

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        if filename:
            return Path(filename).stem.replace("_", " ").replace("-", " ").title()
        return "Document"

    title = None

    # 1. Markdown H1 (# Title) in first 10 lines
    for line in lines[:10]:
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            title = m.group(1).strip()
            break

    # 2. Key-value style headers in first 5 lines (e.g. Title: ..., Scheme Name: ..., etc.)
    if not title:
        header_pattern = r"^(?:title|subject|scheme name|document name|policy name|project|report)\s*[:\-]?\s*(.*)$"
        for idx, line in enumerate(lines[:5]):
            m = re.match(header_pattern, line, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val:
                    title = val
                    break
                # Standalone label on this line, title is on the next line
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    if len(next_line) <= 120 and not next_line.endswith((".", "!", "?", ";")):
                        title = next_line
                        break

    # 3. First line standalone if short and not a full paragraph
    if not title:
        first_line = lines[0]
        if len(first_line) <= 100 and not first_line.endswith((".", "!", "?", ";")):
            words = first_line.split()
            if len(words) <= 12:
                title = first_line

    # 4. Fallback to clean filename stem
    if not title:
        if filename:
            title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        else:
            title = "Document"

    # 5. Acronym / abbreviation enrichment: check if title is defined with acronym in opening text
    # e.g. "Title (ACRONYM)" in opening 1500 characters
    if title:
        m_paren = re.search(rf"{re.escape(title)}\s*\(([^)]+)\)", text[:1500], re.IGNORECASE)
        if m_paren:
            acronym = m_paren.group(1).strip()
            if acronym and acronym not in title:
                title = f"{title} ({acronym})"

    return title


def extract_sections(text: str) -> list[dict]:
    """
    Extract hierarchical sections using multi-signal structural detection.

    Signals evaluated:
    - Markdown headings (#, ##, ###, etc.)
    - Structural prefixes (SECTION, CHAPTER, ARTICLE, PART, etc.)
    - Standalone ALL-CAPS lines (H1 level)
    - Standalone Title-Case lines surrounded by blank lines (H2 level)
    - Distinguishes major headings from numbered sub-items (e.g. '1. Item', '(a) Spec')
      which remain content under their parent heading.
    - Preserves heading hierarchy (heading_path).
    - Preserves headings even if no body text immediately follows (e.g. consecutive headings).

    Args:
        text: Cleaned text string.

    Returns:
        List of dicts: [
            {"section": str, "heading_path": list[str], "content": str},
            ...
        ]
    """
    lines = text.split("\n")
    sections = []
    current_lines = []
    active_hierarchy = {}

    prev_blank = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        next_blank = (i + 1 >= len(lines)) or (not lines[i + 1].strip())

        heading_info = _detect_heading(stripped, prev_blank, next_blank)
        if heading_info is not None:
            title, level = heading_info

            # If we had accumulated content under the previous heading, commit it
            if current_lines and any(l.strip() for l in current_lines):
                content = "\n".join(current_lines).strip()
                path = [active_hierarchy[k] for k in sorted(active_hierarchy) if k in active_hierarchy]
                sec_name = path[-1] if path else "DOCUMENT"
                sections.append({
                    "section": sec_name,
                    "heading_path": path,
                    "content": content,
                })

            current_lines = []
            # Update active hierarchy stack
            active_hierarchy = {k: v for k, v in active_hierarchy.items() if k < level}
            active_hierarchy[level] = title
        else:
            current_lines.append(line)

        prev_blank = not stripped

    # Commit remaining content
    if current_lines and any(l.strip() for l in current_lines):
        content = "\n".join(current_lines).strip()
        path = [active_hierarchy[k] for k in sorted(active_hierarchy) if k in active_hierarchy]
        sec_name = path[-1] if path else "DOCUMENT"
        sections.append({
            "section": sec_name,
            "heading_path": path,
            "content": content,
        })

    if not sections:
        sections = [{
            "section": "DOCUMENT",
            "heading_path": ["DOCUMENT"],
            "content": text.strip()
        }]

    return sections


def _is_sub_item(line: str) -> bool:
    """
    Check if a line is a list item, bullet point, or numbered sub-item.
    Examples:
        '1. Income Category'
        '2) Requirements'
        '(a) Eligibility'
        '- Bullet point'
        '* Feature'
        '• Note'
    These should remain content within the current section rather than
    becoming top-level section breaks.
    """
    pattern = r"^(\d+[\.\)]|\([a-zA-Z0-9]+\)|[a-zA-Z][\.\)]|[-*•])\s+"
    return bool(re.match(pattern, line))


def _detect_heading(line: str, prev_blank: bool, next_blank: bool) -> Optional[tuple[str, int]]:
    """
    Multi-signal heading detector.
    Returns (heading_title, level) or None.

    Level 1: Markdown #, ALL-CAPS standalone lines, major keywords.
    Level 2: Markdown ##, Title-Case standalone lines.
    Level 3+: Markdown ###, ####, etc.
    """
    if not line:
        return None

    if len(line) > 80:
        return None

    # Sentence-ending punctuation disqualifies a line from being a heading
    if line.endswith((".", "!", "?", ",", ";")):
        return None

    # Numbered list items, bullet points, or sub-clauses are content, not headings
    if _is_sub_item(line):
        return None

    # 1. Markdown headings (#, ##, ###)
    md_match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if md_match:
        level = len(md_match.group(1))
        title = md_match.group(2).strip()
        return (title, level)

    # 2. Structural keyword prefixes (e.g. SECTION 1, CHAPTER 2, ARTICLE IV)
    struct_pattern = r"^(SECTION|CHAPTER|ARTICLE|PART|APPENDIX|MODULE|SCHEDULE)\s+[0-9IVXLCDM]+"
    if re.match(struct_pattern, line, re.IGNORECASE):
        return (line, 1)

    alpha_chars = "".join(c for c in line if c.isalpha())
    if not alpha_chars or len(alpha_chars) < 3:
        return None

    words = line.split()

    # 3. ALL-CAPS standalone heading (e.g. ELIGIBILITY, APPLICATION PROCESS)
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio >= 0.85 and len(words) <= 8:
        return (line, 1)

    # 4. Standalone Title-Case heading surrounded by blank lines
    # (e.g. System Overview, General Provisions)
    if prev_blank and next_blank and len(words) <= 6:
        if all(w[0].isupper() for w in words if w[0].isalpha()):
            return (line, 2)

    return None
