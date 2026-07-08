from __future__ import annotations

from dataclasses import dataclass, field
import base64
from io import BytesIO
import hashlib
import ipaddress
import json
import os
import re
import threading
from typing import Iterable, Sequence
from urllib.parse import parse_qsl, urlsplit

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from rapidocr import RapidOCR

from pdf_service import PageView, WordBox, normalise_rect
from feedback_store import apply_local_calibration
from redaction_knowledge import compact_examples, compact_playbook

load_dotenv()

Rect = tuple[float, float, float, float]

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/minimax-m3"

CATEGORY_LABELS: dict[str, str] = {
    "email_address": "Email address",
    "phone_number": "Phone number",
    "home_address": "Home address",
    "username": "Username",
    "full_name": "Full name",
    "account_number": "Account number",
    "bank_card_number": "Bank card-like number",
    "api_key": "API key",
    "access_token": "Access token",
    "password": "Password / password-like value",
    "database_connection_string": "Database connection string",
    "private_key": "Private key",
    "ip_address": "IP address",
    "file_path": "File path",
    "sensitive_url": "URL with sensitive parameters",
    "student_id": "Student ID",
    "employee_id": "Employee ID",
    "date_of_birth": "Date of birth",
    "private_chat": "Private chat message / panel",
    "authentication_code": "Authentication code",
    "qr_code": "QR code",
    "general_url": "Link / URL",
    "person_image": "Person / face image",
    "custom_request": "Custom request",
}

VALID_CATEGORIES = set(CATEGORY_LABELS)
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "code",
    "credential",
    "jwt",
    "key",
    "password",
    "passcode",
    "secret",
    "session",
    "sessionid",
    "sig",
    "signature",
    "token",
}


@dataclass(frozen=True)
class AnalysisToken:
    token_id: str
    page_index: int
    text: str
    rect: Rect
    confidence: float = 1.0
    source: str = "embedded_text"


@dataclass
class SensitiveSuggestion:
    suggestion_id: str
    page_index: int
    category: str
    confidence: float
    rects: list[Rect]
    preview: str
    reason: str
    source: str
    token_ids: tuple[str, ...] = field(default_factory=tuple)
    selected: bool = True
    applied: bool = False

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category.replace("_", " ").title())


@dataclass(frozen=True)
class AnalysisResult:
    suggestions: list[SensitiveSuggestion]
    warnings: tuple[str, ...]
    token_count: int
    ai_used: bool


@dataclass(frozen=True)
class _TokenLine:
    tokens: tuple[AnalysisToken, ...]
    text: str
    spans: tuple[tuple[int, int, AnalysisToken], ...]


class _VisionFinding(BaseModel):
    category: str
    confidence: float
    token_ids: list[str] = []
    bbox: list[float] = []
    preview: str = ""
    reason: str = ""


class _VisionResponse(BaseModel):
    findings: list[_VisionFinding]


_OCR_ENGINE: RapidOCR | None = None
_OCR_LOCK = threading.Lock()


def _get_ocr_engine() -> RapidOCR:
    global _OCR_ENGINE
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE


def embedded_tokens(words: Sequence[WordBox], page_index: int) -> list[AnalysisToken]:
    return [
        AnalysisToken(
            token_id=f"p{page_index}_e{index}",
            page_index=page_index,
            text=word.text,
            rect=normalise_rect(word.rect),
            confidence=1.0,
            source="embedded_text",
        )
        for index, word in enumerate(words)
        if word.text.strip()
    ]


def _quad_to_rect(points: Iterable[Iterable[float]]) -> Rect:
    values = [(float(point[0]), float(point[1])) for point in points]
    xs = [value[0] for value in values]
    ys = [value[1] for value in values]
    return min(xs), min(ys), max(xs), max(ys)


def _canvas_rect_to_page(rect: Rect, view: PageView) -> Rect | None:
    x0, y0, x1, y1 = normalise_rect(rect)
    centre_x = (x0 + x1) / 2
    centre_y = (y0 + y1) / 2
    if not view.contains_canvas_point(centre_x, centre_y):
        return None
    page_x0, page_y0 = view.canvas_to_page(x0, y0)
    page_x1, page_y1 = view.canvas_to_page(x1, y1)
    converted = normalise_rect((page_x0, page_y0, page_x1, page_y1))
    if converted[2] - converted[0] < 0.5 or converted[3] - converted[1] < 0.5:
        return None
    return converted


def ocr_tokens(image: Image.Image, view: PageView, page_index: int) -> list[AnalysisToken]:
    """Run local OCR and return word-level boxes in page/image coordinates."""

    rgb = np.asarray(image.convert("RGB"))
    engine = _get_ocr_engine()
    with _OCR_LOCK:
        output = engine(rgb, return_word_box=True)

    tokens: list[AnalysisToken] = []
    word_results = getattr(output, "word_results", None)
    if word_results:
        index = 0
        for line_words in word_results:
            for word_result in line_words:
                if len(word_result) < 3:
                    continue
                text = str(word_result[0]).strip()
                if not text:
                    continue
                score = float(word_result[1])
                page_rect = _canvas_rect_to_page(_quad_to_rect(word_result[2]), view)
                if page_rect is None:
                    continue
                tokens.append(
                    AnalysisToken(
                        token_id=f"p{page_index}_o{index}",
                        page_index=page_index,
                        text=text,
                        rect=page_rect,
                        confidence=max(0.0, min(score, 1.0)),
                        source="ocr",
                    )
                )
                index += 1
        return tokens

    boxes = getattr(output, "boxes", None)
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if boxes is None:
        boxes = []
    if texts is None:
        texts = []
    if scores is None:
        scores = []
    for index, (box, text, score) in enumerate(zip(boxes, texts, scores)):
        text = str(text).strip()
        if not text:
            continue
        page_rect = _canvas_rect_to_page(_quad_to_rect(box), view)
        if page_rect is None:
            continue
        tokens.append(
            AnalysisToken(
                token_id=f"p{page_index}_o{index}",
                page_index=page_index,
                text=text,
                rect=page_rect,
                confidence=max(0.0, min(float(score), 1.0)),
                source="ocr",
            )
        )
    return tokens


def _rect_intersection_ratio(first: Rect, second: Rect) -> float:
    ax0, ay0, ax1, ay1 = normalise_rect(first)
    bx0, by0, bx1, by1 = normalise_rect(second)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return intersection / smaller if smaller > 0 else 0.0


def _normalised_text(value: str) -> str:
    return re.sub(r"\W+", "", value).lower()


def merge_embedded_and_ocr(
    embedded: Sequence[AnalysisToken], ocr: Sequence[AnalysisToken]
) -> list[AnalysisToken]:
    if not embedded:
        return list(ocr)
    merged = list(embedded)
    for candidate in ocr:
        duplicate = any(
            _rect_intersection_ratio(candidate.rect, existing.rect) >= 0.65
            and (
                _normalised_text(candidate.text) == _normalised_text(existing.text)
                or _rect_intersection_ratio(candidate.rect, existing.rect) >= 0.88
            )
            for existing in embedded
        )
        if not duplicate:
            merged.append(candidate)
    return merged


def detect_qr_codes(image: Image.Image, view: PageView, page_index: int) -> list[SensitiveSuggestion]:
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    detected: list[tuple[Rect, bool]] = []

    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(bgr)
        if ok and points is not None:
            for index, quad in enumerate(points):
                decoded = bool(decoded_info[index]) if index < len(decoded_info) else False
                detected.append((_quad_to_rect(quad), decoded))
    except cv2.error:
        pass

    if not detected:
        try:
            decoded, points, _ = detector.detectAndDecode(bgr)
            if points is not None:
                detected.append((_quad_to_rect(points), bool(decoded)))
        except cv2.error:
            pass

    suggestions: list[SensitiveSuggestion] = []
    for rect, decoded in detected:
        page_rect = _canvas_rect_to_page(rect, view)
        if page_rect is None:
            continue
        confidence = 0.99 if decoded else 0.91
        suggestions.append(
            _make_suggestion(
                page_index=page_index,
                category="qr_code",
                confidence=confidence,
                rects=[_pad_rect(page_rect, 2.0, view)],
                raw_preview="QR code",
                reason="A QR code was detected locally; its payload may contain private data or authentication links.",
                source="QR detector",
            )
        )
    return suggestions


def _line_threshold(token: AnalysisToken) -> float:
    return max(3.0, (token.rect[3] - token.rect[1]) * 0.65)


def _build_lines(tokens: Sequence[AnalysisToken]) -> list[_TokenLine]:
    ordered = sorted(tokens, key=lambda token: ((token.rect[1] + token.rect[3]) / 2, token.rect[0]))
    groups: list[list[AnalysisToken]] = []
    centres: list[float] = []

    for token in ordered:
        centre = (token.rect[1] + token.rect[3]) / 2
        best_index: int | None = None
        best_distance = float("inf")
        for index, line_centre in enumerate(centres):
            distance = abs(centre - line_centre)
            if distance <= _line_threshold(token) and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            groups.append([token])
            centres.append(centre)
        else:
            groups[best_index].append(token)
            centres[best_index] = sum((item.rect[1] + item.rect[3]) / 2 for item in groups[best_index]) / len(
                groups[best_index]
            )

    lines: list[_TokenLine] = []
    for group in groups:
        sorted_group = sorted(group, key=lambda token: token.rect[0])
        text_parts: list[str] = []
        spans: list[tuple[int, int, AnalysisToken]] = []
        cursor = 0
        for index, token in enumerate(sorted_group):
            if index:
                text_parts.append(" ")
                cursor += 1
            start = cursor
            text_parts.append(token.text)
            cursor += len(token.text)
            spans.append((start, cursor, token))
        lines.append(_TokenLine(tuple(sorted_group), "".join(text_parts), tuple(spans)))

    return sorted(lines, key=lambda line: min(token.rect[1] for token in line.tokens))


def _tokens_for_span(line: _TokenLine, start: int, end: int) -> list[AnalysisToken]:
    return [token for token_start, token_end, token in line.spans if token_start < end and token_end > start]


def _rects_for_tokens(tokens: Sequence[AnalysisToken]) -> list[Rect]:
    return [normalise_rect(token.rect) for token in tokens]


def _confidence_for_tokens(base: float, tokens: Sequence[AnalysisToken]) -> float:
    if not tokens:
        return base
    token_quality = sum(token.confidence for token in tokens) / len(tokens)
    return max(0.0, min(0.995, base * 0.82 + token_quality * 0.18))


def _mask_preview(text: str, category: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return CATEGORY_LABELS.get(category, category)
    if category in {"api_key", "access_token", "password", "database_connection_string", "private_key"}:
        if len(clean) <= 8:
            return "•" * len(clean)
        return f"{clean[:3]}••••••{clean[-3:]}"
    if category in {"bank_card_number", "account_number", "authentication_code"}:
        digits = re.sub(r"\D", "", clean)
        return f"•••• {digits[-4:]}" if len(digits) >= 4 else "••••"
    if len(clean) > 72:
        return clean[:69] + "…"
    return clean


def _suggestion_id(page_index: int, category: str, rects: Sequence[Rect], preview: str) -> str:
    payload = json.dumps(
        [page_index, category, [[round(value, 2) for value in rect] for rect in rects], preview],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _make_suggestion(
    *,
    page_index: int,
    category: str,
    confidence: float,
    rects: Sequence[Rect],
    raw_preview: str,
    reason: str,
    source: str,
    token_ids: Sequence[str] = (),
) -> SensitiveSuggestion:
    normalised_rects = [normalise_rect(rect) for rect in rects]
    preview = _mask_preview(raw_preview, category)
    return SensitiveSuggestion(
        suggestion_id=_suggestion_id(page_index, category, normalised_rects, preview),
        page_index=page_index,
        category=category,
        confidence=max(0.0, min(float(confidence), 0.995)),
        rects=normalised_rects,
        preview=preview,
        reason=reason,
        source=source,
        token_ids=tuple(dict.fromkeys(token_ids)),
    )


def _add_regex_match(
    suggestions: list[SensitiveSuggestion],
    *,
    line: _TokenLine,
    match: re.Match[str],
    page_index: int,
    category: str,
    base_confidence: float,
    reason: str,
    group: int = 0,
) -> None:
    start, end = match.span(group)
    matched_tokens = _tokens_for_span(line, start, end)
    if not matched_tokens:
        return
    suggestions.append(
        _make_suggestion(
            page_index=page_index,
            category=category,
            confidence=_confidence_for_tokens(base_confidence, matched_tokens),
            rects=_rects_for_tokens(matched_tokens),
            raw_preview=match.group(group),
            reason=reason,
            source="Pattern detector",
            token_ids=[token.token_id for token in matched_tokens],
        )
    )


def _luhn_valid(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _valid_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 8 <= len(digits) <= 15 and not (len(digits) >= 13 and _luhn_valid(digits))


def _valid_sensitive_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.rstrip(".,);]"))
    except ValueError:
        return False
    return any(key.lower() in SENSITIVE_QUERY_KEYS for key, _ in parse_qsl(parsed.query, keep_blank_values=True))


def _valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip("[](),.;"))
        return True
    except ValueError:
        return False


NAME_WORD_RE = r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*"
NAME_VALUE_RE = rf"{NAME_WORD_RE}(?:\s+{NAME_WORD_RE}){{1,4}}"
NAME_LABEL_RE = re.compile(
    rf"(?i)\b(?:full\s+name|applicant(?:'s)?\s+name|candidate(?:'s)?\s+name|student\s+name|"
    rf"employee\s+name|customer\s+name|account\s+holder(?:\s+name)?|patient\s+name|tenant\s+name|"
    rf"contact\s+name|name)\b\s*[:#=-]?\s*({NAME_VALUE_RE})\s*$"
)
ADDRESS_LABEL_RE = re.compile(
    r"(?i)^\s*(?:home|residential|postal|mailing|billing|delivery|correspondence|current|permanent)?\s*address\b\s*[:#=-]?\s*(.*)$"
)
UK_POSTCODE_RE = re.compile(r"(?i)\b(?:GIR\s?0AA|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b")
US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
STREET_RE = re.compile(
    r"(?i)\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|close|court|ct\.?|"
    r"way|crescent|gardens?|grove|place|square|terrace|park|hill|mews|walk|row|boulevard|blvd\.?|"
    r"highway|hwy\.?|rise|view|gate|bank|quay)\b"
)
ADDRESS_STOP_LABEL_RE = re.compile(
    r"(?i)^\s*(?:email|e-mail|phone|telephone|mobile|dob|date\s+of\s+birth|student\s+id|employee\s+id|"
    r"account|username|password|nationality|gender|company|organisation|department)\b\s*[:#=-]"
)
NAME_HEADING_STOP_WORDS = {
    "address", "application", "applicant", "candidate", "company", "contact", "curriculum", "details",
    "document", "education", "electrical", "employee", "engineer", "engineering", "experience", "invoice",
    "manager", "personal", "profile", "project", "report", "resume", "statement", "student", "summary",
    "university", "vitae", "work",
}


def _tokens_text(tokens: Sequence[AnalysisToken]) -> str:
    return " ".join(token.text for token in tokens).strip()


def _looks_like_person_name(value: str) -> bool:
    value = re.sub(r"\s+", " ", value).strip(" ,:;-\t")
    words = value.split()
    if not 2 <= len(words) <= 5 or any(any(char.isdigit() for char in word) for word in words):
        return False
    if any(word.lower().strip(".'’-") in NAME_HEADING_STOP_WORDS for word in words):
        return False
    for word in words:
        clean = word.strip(".'’- ")
        if len(clean) < 2 or not re.fullmatch(NAME_WORD_RE, clean):
            return False
        if not (clean[0].isupper() or clean.isupper()):
            return False
    return True


def _add_token_suggestion(
    suggestions: list[SensitiveSuggestion],
    *,
    page_index: int,
    category: str,
    tokens: Sequence[AnalysisToken],
    confidence: float,
    reason: str,
    source: str = "Context detector",
) -> None:
    if not tokens:
        return
    suggestions.append(
        _make_suggestion(
            page_index=page_index,
            category=category,
            confidence=_confidence_for_tokens(confidence, tokens),
            rects=_rects_for_tokens(tokens),
            raw_preview=_tokens_text(tokens),
            reason=reason,
            source=source,
            token_ids=[token.token_id for token in tokens],
        )
    )


def contextual_name_and_address_suggestions(
    lines: Sequence[_TokenLine], page_index: int
) -> list[SensitiveSuggestion]:
    """Detect labelled names and postal addresses without relying on the VLM.

    These are deliberately review suggestions, not automatic redactions. The labelled
    cases are high-confidence; unlabelled document-header names and street addresses
    are lower-confidence so the user can approve or reject them in the sidebar.
    """

    suggestions: list[SensitiveSuggestion] = []
    if not lines:
        return suggestions

    # Labelled full names, e.g. "Full name: Richmond Kyawzay".
    for line in lines:
        match = NAME_LABEL_RE.search(line.text)
        if not match or not _looks_like_person_name(match.group(1)):
            continue
        tokens = _tokens_for_span(line, *match.span(1))
        _add_token_suggestion(
            suggestions,
            page_index=page_index,
            category="full_name",
            tokens=tokens,
            confidence=0.965,
            reason="A complete personal name appears next to a name-related field label.",
        )

    # A prominent title-cased line near the top of a CV, application or profile is
    # often the document owner's name. Requiring larger-than-median text reduces
    # false positives from ordinary headings.
    token_heights = [token.rect[3] - token.rect[1] for line in lines for token in line.tokens]
    median_height = sorted(token_heights)[len(token_heights) // 2] if token_heights else 0.0
    for line in lines[:6]:
        candidate = line.text.strip()
        line_height = sum(token.rect[3] - token.rect[1] for token in line.tokens) / max(len(line.tokens), 1)
        if not _looks_like_person_name(candidate):
            continue
        is_prominent = line_height >= max(median_height * 1.10, median_height + 0.5)
        is_very_early = lines.index(line) <= 2
        if is_prominent or is_very_early:
            _add_token_suggestion(
                suggestions,
                page_index=page_index,
                category="full_name",
                tokens=line.tokens,
                confidence=0.76 if is_prominent else 0.68,
                reason=(
                    "A prominent personal-name-shaped line appears near the top of the document."
                    if is_prominent
                    else "A personal-name-shaped line appears in the document header; it is offered as a review suggestion."
                ),
            )

    consumed_address_lines: set[int] = set()

    # Labelled addresses may continue over several lines. Include lines until a
    # postcode/ZIP is reached or another labelled field begins.
    for index, line in enumerate(lines):
        label_match = ADDRESS_LABEL_RE.match(line.text)
        if not label_match:
            continue
        address_tokens = list(_tokens_for_span(line, *label_match.span(1))) if label_match.group(1).strip() else []
        used_indexes = {index}
        for next_index in range(index + 1, min(index + 5, len(lines))):
            next_line = lines[next_index]
            if ADDRESS_STOP_LABEL_RE.match(next_line.text):
                break
            address_tokens.extend(next_line.tokens)
            used_indexes.add(next_index)
            combined = _tokens_text(address_tokens)
            if UK_POSTCODE_RE.search(combined) or US_ZIP_RE.search(combined):
                break
        combined = _tokens_text(address_tokens)
        if address_tokens and (
            STREET_RE.search(combined)
            or UK_POSTCODE_RE.search(combined)
            or US_ZIP_RE.search(combined)
            or re.search(r"\b\d{1,5}[A-Za-z]?\b", combined)
        ):
            _add_token_suggestion(
                suggestions,
                page_index=page_index,
                category="home_address",
                tokens=address_tokens,
                confidence=0.965,
                reason="A postal or residential address appears next to an address field label.",
            )
            consumed_address_lines.update(used_indexes)

    # Unlabelled postal addresses: start at a house number/name + street suffix and
    # include locality/postcode lines immediately beneath it.
    for index, line in enumerate(lines):
        if index in consumed_address_lines:
            continue
        text = line.text.strip()
        starts_like_property = bool(re.match(r"(?i)^(?:flat|apartment|apt\.?|unit|suite)?\s*\d{1,5}[A-Za-z]?(?:[-/]\d{1,5}[A-Za-z]?)?\b", text))
        if not (starts_like_property and STREET_RE.search(text)):
            continue
        address_tokens = list(line.tokens)
        used_indexes = {index}
        for next_index in range(index + 1, min(index + 4, len(lines))):
            next_line = lines[next_index]
            if ADDRESS_STOP_LABEL_RE.match(next_line.text):
                break
            # Only attach nearby lines; the line builder is ordered vertically.
            vertical_gap = min(token.rect[1] for token in next_line.tokens) - max(token.rect[3] for token in lines[next_index - 1].tokens)
            typical_height = max(token.rect[3] - token.rect[1] for token in line.tokens)
            if vertical_gap > max(typical_height * 1.8, 18.0):
                break
            address_tokens.extend(next_line.tokens)
            used_indexes.add(next_index)
            combined = _tokens_text(address_tokens)
            if UK_POSTCODE_RE.search(combined) or US_ZIP_RE.search(combined):
                break
        _add_token_suggestion(
            suggestions,
            page_index=page_index,
            category="home_address",
            tokens=address_tokens,
            confidence=0.87 if (UK_POSTCODE_RE.search(_tokens_text(address_tokens)) or US_ZIP_RE.search(_tokens_text(address_tokens))) else 0.81,
            reason="The text has the structure of a street address, with an optional locality or postcode.",
        )
        consumed_address_lines.update(used_indexes)

    # Catch a postcode line whose street line is directly above it.
    for index, line in enumerate(lines):
        if index in consumed_address_lines or not (UK_POSTCODE_RE.search(line.text) or US_ZIP_RE.search(line.text)):
            continue
        start = max(0, index - 2)
        candidate_lines = lines[start : index + 1]
        combined = " ".join(item.text for item in candidate_lines)
        if STREET_RE.search(combined):
            address_tokens = [token for item in candidate_lines for token in item.tokens]
            _add_token_suggestion(
                suggestions,
                page_index=page_index,
                category="home_address",
                tokens=address_tokens,
                confidence=0.89,
                reason="A street line and postal code together form a likely postal address.",
            )

    return suggestions


def local_pattern_suggestions(tokens: Sequence[AnalysisToken], page_index: int) -> list[SensitiveSuggestion]:
    lines = _build_lines(tokens)
    suggestions: list[SensitiveSuggestion] = contextual_name_and_address_suggestions(lines, page_index)

    simple_patterns: list[tuple[str, re.Pattern[str], float, str, int]] = [
        (
            "email_address",
            re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
            0.99,
            "The text matches an email-address pattern.",
            0,
        ),
        (
            "database_connection_string",
            re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|mssql|jdbc:[a-z0-9_-]+)://[^\s<>\"']+"),
            0.995,
            "The value resembles a database connection string and may include credentials.",
            0,
        ),
        (
            "access_token",
            re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
            0.995,
            "The value has the structure of a JSON Web Token.",
            0,
        ),
        (
            "api_key",
            re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|fw_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,})\b"),
            0.995,
            "The value matches a common API-key or access-token format.",
            0,
        ),
        (
            "file_path",
            re.compile(r"(?i)\b[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"),
            0.93,
            "The text resembles a Windows file-system path.",
            0,
        ),
        (
            "file_path",
            re.compile(r"(?<!\w)/(?:home|Users|var|etc|opt|srv|tmp|mnt|root|usr|app)/[^\s<>\"']+"),
            0.92,
            "The text resembles an absolute Unix-like file-system path.",
            0,
        ),
        (
            "student_id",
            re.compile(r"(?i)\b(?:student(?:\s+(?:id|number))|university\s+id)\b\s*[:#=-]?\s*([A-Z0-9-]{4,24})"),
            0.96,
            "A value appears next to a student-ID label.",
            1,
        ),
        (
            "employee_id",
            re.compile(r"(?i)\b(?:employee(?:\s+(?:id|number))|staff\s+id|personnel\s+number)\b\s*[:#=-]?\s*([A-Z0-9-]{4,24})"),
            0.96,
            "A value appears next to an employee-ID label.",
            1,
        ),
        (
            "username",
            re.compile(r"(?i)\b(?:username|user\s*name|handle|login)\b\s*[:=]\s*(@?[A-Za-z0-9_.-]{3,64})"),
            0.95,
            "A value appears next to a username or login label.",
            1,
        ),
        (
            "date_of_birth",
            re.compile(r"(?i)\b(?:date\s+of\s+birth|d\.o\.b\.?|dob|born)\b\s*[:#=-]?\s*((?:\d{1,2}[./-]){2}\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|(?:\d{1,2}\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4})"),
            0.97,
            "A date appears next to a date-of-birth label.",
            1,
        ),
        (
            "authentication_code",
            re.compile(r"(?i)\b(?:otp|one[- ]time(?:\s+(?:password|code))?|verification\s+code|authentication\s+code|2fa\s+code|mfa\s+code)\b\s*[:#=-]?\s*(\d{4,8})"),
            0.99,
            "A short numeric value appears next to an authentication-code label.",
            1,
        ),
        (
            "account_number",
            re.compile(r"(?i)\b(?:account(?:\s+(?:number|no\.?))?|sort\s+code|iban)\b\s*[:#=-]?\s*([A-Z0-9][A-Z0-9 -]{5,33})"),
            0.94,
            "A value appears next to a bank-account or IBAN label.",
            1,
        ),
        (
            "password",
            re.compile(r"(?i)\b(?:password|passwd|pwd|passcode)\b\s*[:=]\s*([^\s,;]{4,128})"),
            0.995,
            "A value appears next to a password-like label.",
            1,
        ),
        (
            "api_key",
            re.compile(r"(?i)\b(?:api[_ -]?key|client[_ -]?secret)\b\s*[:=]\s*([^\s,;]{8,256})"),
            0.99,
            "A value appears next to an API-key or client-secret label.",
            1,
        ),
        (
            "access_token",
            re.compile(r"(?i)\b(?:access[_ -]?token|bearer[_ -]?token|refresh[_ -]?token|auth[_ -]?token)\b\s*[:=]\s*([^\s,;]{8,512})"),
            0.99,
            "A value appears next to an access-token label.",
            1,
        ),
    ]

    phone_pattern = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")
    url_pattern = re.compile(r"(?i)https?://[^\s<>\"']+")
    ipv4_pattern = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
    ipv6_pattern = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])")
    card_pattern = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

    for line in lines:
        for category, pattern, confidence, reason, group in simple_patterns:
            for match in pattern.finditer(line.text):
                _add_regex_match(
                    suggestions,
                    line=line,
                    match=match,
                    page_index=page_index,
                    category=category,
                    base_confidence=confidence,
                    reason=reason,
                    group=group,
                )

        for match in phone_pattern.finditer(line.text):
            if _valid_phone(match.group(0)):
                _add_regex_match(
                    suggestions,
                    line=line,
                    match=match,
                    page_index=page_index,
                    category="phone_number",
                    base_confidence=0.91,
                    reason="The value resembles an international or local phone number.",
                )

        for match in url_pattern.finditer(line.text):
            if _valid_sensitive_url(match.group(0)):
                _add_regex_match(
                    suggestions,
                    line=line,
                    match=match,
                    page_index=page_index,
                    category="sensitive_url",
                    base_confidence=0.98,
                    reason="The URL contains a query parameter commonly used for credentials, sessions or access tokens.",
                )

        for pattern in (ipv4_pattern, ipv6_pattern):
            for match in pattern.finditer(line.text):
                if _valid_ip(match.group(0)):
                    _add_regex_match(
                        suggestions,
                        line=line,
                        match=match,
                        page_index=page_index,
                        category="ip_address",
                        base_confidence=0.96,
                        reason="The value is a valid IP address.",
                    )

        for match in card_pattern.finditer(line.text):
            if _luhn_valid(match.group(0)):
                _add_regex_match(
                    suggestions,
                    line=line,
                    match=match,
                    page_index=page_index,
                    category="bank_card_number",
                    base_confidence=0.985,
                    reason="The digit sequence has a bank-card-like length and passes the Luhn checksum.",
                )

    # PEM private-key blocks may span many OCR/text lines.
    start_pattern = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
    end_pattern = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
    index = 0
    while index < len(lines):
        if not start_pattern.search(lines[index].text):
            index += 1
            continue
        block_lines = [lines[index]]
        cursor = index + 1
        while cursor < len(lines) and cursor <= index + 80:
            block_lines.append(lines[cursor])
            if end_pattern.search(lines[cursor].text):
                break
            cursor += 1
        block_tokens = [token for line in block_lines for token in line.tokens]
        suggestions.append(
            _make_suggestion(
                page_index=page_index,
                category="private_key",
                confidence=_confidence_for_tokens(0.995, block_tokens),
                rects=_rects_for_tokens(block_tokens),
                raw_preview="Private key block",
                reason="A PEM private-key block marker was detected.",
                source="Pattern detector",
                token_ids=[token.token_id for token in block_tokens],
            )
        )
        index = max(cursor + 1, index + 1)

    return suggestions


def _ai_prompt_payload(lines: Sequence[_TokenLine]) -> str:
    payload = []
    for line_index, line in enumerate(lines):
        payload.append(
            {
                "line_id": f"L{line_index}",
                "tokens": [{"id": token.token_id, "text": token.text} for token in line.tokens],
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _chunk_lines(lines: Sequence[_TokenLine], max_chars: int = 12_000, max_lines: int = 100) -> list[list[_TokenLine]]:
    chunks: list[list[_TokenLine]] = []
    current: list[_TokenLine] = []
    current_chars = 0
    for line in lines:
        line_chars = len(line.text) + sum(len(token.token_id) + 6 for token in line.tokens)
        if current and (len(current) >= max_lines or current_chars + line_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_chars
    if current:
        chunks.append(current)
    return chunks


def _model_schema() -> dict:
    return _VisionResponse.model_json_schema()


def _document_crop(image: Image.Image, view: PageView) -> Image.Image:
    """Remove the grey editor margin before sending the page to the VLM."""

    left = max(0, round(view.offset_x))
    top = max(0, round(view.offset_y))
    right = min(image.width, round(view.offset_x + view.pixel_width))
    bottom = min(image.height, round(view.offset_y + view.pixel_height))
    if right <= left or bottom <= top:
        return image.convert("RGB")
    return image.crop((left, top, right, bottom)).convert("RGB")


def _image_data_url(image: Image.Image) -> str:
    """Encode a downsized page as a base64 JPEG accepted by Fireworks vision models."""

    page_image = image.copy()
    maximum_side = int(os.getenv("FIREWORKS_IMAGE_MAX_SIDE", "1600"))
    if maximum_side > 0 and max(page_image.size) > maximum_side:
        scale = maximum_side / max(page_image.size)
        page_image = page_image.resize(
            (max(1, round(page_image.width * scale)), max(1, round(page_image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    buffer = BytesIO()
    page_image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _bbox_to_page_rect(bbox: Sequence[float], view: PageView) -> Rect | None:
    """Convert a VLM bbox in the 0..1000 coordinate system into page coordinates."""

    if len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (x0, y0, x1, y1)):
        return None

    x0, y0, x1, y1 = normalise_rect((x0, y0, x1, y1))
    x0 = min(max(x0, 0.0), 1000.0)
    y0 = min(max(y0, 0.0), 1000.0)
    x1 = min(max(x1, 0.0), 1000.0)
    y1 = min(max(y1, 0.0), 1000.0)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    return normalise_rect(
        (
            x0 / 1000.0 * view.page_width,
            y0 / 1000.0 * view.page_height,
            x1 / 1000.0 * view.page_width,
            y1 / 1000.0 * view.page_height,
        )
    )


def _extract_json(content: str) -> str:
    clean = content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    first = clean.find("{")
    last = clean.rfind("}")
    if first >= 0 and last > first:
        return clean[first : last + 1]
    return clean


def _vision_prompt_payload(tokens: Sequence[AnalysisToken]) -> str:
    lines = _build_lines(tokens)
    payload = []
    for line_index, line in enumerate(lines):
        payload.append(
            {
                "line_id": f"L{line_index}",
                "tokens": [{"id": token.token_id, "text": token.text} for token in line.tokens],
            }
        )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))



def _custom_prompt_requests_links(custom_instruction: str) -> bool:
    text = custom_instruction.lower()
    return any(phrase in text for phrase in (
        "all links", "every link", "web links", "urls", "url", "hyperlinks", "website links"
    ))


def _custom_prompt_requests_people(custom_instruction: str) -> bool:
    text = custom_instruction.lower()
    return any(phrase in text for phrase in (
        "pictures of people", "photos of people", "images of people", "people in photos",
        "faces", "face", "portraits", "profile photos", "person images"
    ))


def custom_prompt_local_suggestions(
    *,
    custom_instruction: str,
    tokens: Sequence[AnalysisToken],
    image: Image.Image,
    view: PageView,
    page_index: int,
) -> list[SensitiveSuggestion]:
    """Add deterministic helpers for common custom requests.

    The VLM still receives the full instruction for arbitrary targets. These local
    helpers make requests for all links and human faces more reliable.
    """

    instruction = custom_instruction.strip()
    if not instruction:
        return []

    suggestions: list[SensitiveSuggestion] = []
    if _custom_prompt_requests_links(instruction):
        url_pattern = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\"']+")
        for line in _build_lines(tokens):
            for match in url_pattern.finditer(line.text):
                _add_regex_match(
                    suggestions,
                    line=line,
                    match=match,
                    page_index=page_index,
                    category="general_url",
                    base_confidence=0.985,
                    reason="The user explicitly requested that visible links be redacted.",
                )

    if _custom_prompt_requests_people(instruction):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if not detector.empty():
                rgb = np.asarray(image.convert("RGB"))
                grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                faces = detector.detectMultiScale(
                    grey,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=(32, 32),
                )
                for x, y, width, height in faces:
                    page_rect = _canvas_rect_to_page(
                        (float(x), float(y), float(x + width), float(y + height)), view
                    )
                    if page_rect is None:
                        continue
                    suggestions.append(
                        _make_suggestion(
                            page_index=page_index,
                            category="person_image",
                            confidence=0.84,
                            rects=[_pad_rect(page_rect, 5.0, view)],
                            raw_preview="Detected face / portrait",
                            reason="The user asked to redact pictures of people and a face-like region was detected locally.",
                            source="OpenCV face detector",
                        )
                    )
        except Exception:
            # The Fireworks vision model still handles this custom request.
            pass

    return suggestions


def _call_fireworks_vision(
    *,
    image: Image.Image,
    view: PageView,
    tokens: Sequence[AnalysisToken],
    page_index: int,
    custom_instruction: str = "",
) -> tuple[list[SensitiveSuggestion], list[str], bool]:
    api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        return [], ["FIREWORKS_API_KEY is not configured, so only local pattern, OCR and QR detectors were used."], False

    model = (
        os.getenv("FIREWORKS_VISION_MODEL", "").strip()
        or os.getenv("FIREWORKS_MODEL", "").strip()
        or DEFAULT_FIREWORKS_MODEL
    )
    client = OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)
    token_lookup = {token.token_id: token for token in tokens}
    page_image = _document_crop(image, view)
    data_url = _image_data_url(page_image)

    custom_instruction = custom_instruction.strip()
    custom_rule = (
        "\nCUSTOM REDACTION INSTRUCTION FROM THE USER: " + custom_instruction +
        "\nTreat this instruction as an additional target. Use general_url for ordinary links, "
        "person_image for photographs/faces/portraits, or custom_request for other requested targets. "
        "Do not ignore the standard privacy categories.\n"
        if custom_instruction
        else ""
    )
    system_prompt = f"""You are a high-recall visual privacy detector for a human-reviewed redaction application.
Inspect the document or screenshot image and return JSON only. Identify likely sensitive information, but do not redact anything yourself. The user will approve or reject every suggestion, so prefer a calibrated suggestion over silently missing plausible private data.
Allowed categories: email_address, phone_number, home_address, username, full_name, account_number, bank_card_number, api_key, access_token, password, database_connection_string, private_key, ip_address, file_path, sensitive_url, student_id, employee_id, date_of_birth, private_chat, authentication_code, qr_code, general_url, person_image, custom_request.

Output rules:
- Return one finding per sensitive value or coherent private visual region.
- confidence must be a calibrated probability from 0 to 1. Omit weak guesses below 0.55.
- token_ids must contain only IDs from the supplied token list. Use them whenever the target corresponds to extracted text because they provide exact word boxes.
- bbox is [x0,y0,x1,y1] in a 0..1000 coordinate system relative to the supplied image. Use bbox for visual regions, scanned text not represented by tokens, QR codes, faces/photos, private chat panels, signatures, or when token boxes are insufficient.
- A finding may contain both token_ids and bbox. Do not invent token IDs.
- Select the sensitive value rather than harmless field labels, except private_chat where a coherent message bubble or panel can be selected.
- full_name: include complete personal names in forms, CVs, letters, account pages, messages, signatures and profile headers. Do not return company, product or organisation names.
- home_address: include house/flat, street, locality/city and postcode where visible; return one coherent finding.
- sensitive_url is for links carrying credentials, tokens, sessions, signatures or one-time codes. general_url is for ordinary links only when requested.
- private_chat should cover actual private message content or the coherent message panel.
- qr_code should locate every visible QR code even when its payload cannot be decoded.
- person_image should cover the face or portrait/photo region requested by the user.
- preview must be short and mask secrets where possible.
- reason must be concise.

DETECTION_PLAYBOOK_JSON={compact_playbook()}
SYNTHETIC_FEW_SHOT_EXAMPLES_JSON={compact_examples()}
{custom_rule}
The response must match the supplied JSON schema."""

    text_payload = _vision_prompt_payload(tokens)
    user_text = (
        "Review this page for sensitive information. The optional OCR/embedded-text tokens below may help you "
        "return exact token IDs; also inspect the image itself for visual content and regions not present in OCR.\n\n"
        + (f"CUSTOM_INSTRUCTION={custom_instruction}\n\n" if custom_instruction else "")
        + f"TOKENS_JSON={text_payload}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    parsed: _VisionResponse | None = None
    errors: list[str] = []
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "SensitiveVisualFindings", "schema": _model_schema()},
            },
            temperature=0,
            max_tokens=5000,
            timeout=120,
        )
        content = response.choices[0].message.content or '{"findings":[]}'
        parsed = _VisionResponse.model_validate_json(_extract_json(content))
    except Exception as schema_exc:
        errors.append(f"schema request: {schema_exc}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=5000,
                timeout=120,
            )
            content = response.choices[0].message.content or '{"findings":[]}'
            parsed = _VisionResponse.model_validate_json(_extract_json(content))
        except Exception as json_exc:
            errors.append(f"JSON request: {json_exc}")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=5000,
                    timeout=120,
                )
                content = response.choices[0].message.content or '{"findings":[]}'
                parsed = _VisionResponse.model_validate_json(_extract_json(content))
            except Exception as plain_exc:
                errors.append(f"plain request: {plain_exc}")

    if parsed is None:
        return [], [f"Fireworks vision analysis failed on page {page_index + 1}: {'; '.join(errors)}"], False

    findings: list[SensitiveSuggestion] = []
    model_name = model.rsplit("/", 1)[-1]
    for model_finding in parsed.findings:
        category = model_finding.category.strip().lower()
        if category not in VALID_CATEGORIES:
            continue

        confidence = max(0.0, min(float(model_finding.confidence), 0.995))
        if confidence < 0.55:
            continue

        matched_tokens = [
            token_lookup[token_id]
            for token_id in model_finding.token_ids
            if token_id in token_lookup
        ]
        unique_tokens = list({token.token_id: token for token in matched_tokens}.values())
        rects = _rects_for_tokens(unique_tokens)
        bbox_rect = _bbox_to_page_rect(model_finding.bbox, view)
        if bbox_rect is not None:
            rects.append(bbox_rect)
        if not rects:
            continue

        if unique_tokens:
            confidence = _confidence_for_tokens(confidence, unique_tokens)
        raw_preview = model_finding.preview.strip()
        if not raw_preview and unique_tokens:
            raw_preview = " ".join(
                token.text for token in sorted(unique_tokens, key=lambda item: (item.rect[1], item.rect[0]))
            )
        if not raw_preview:
            raw_preview = CATEGORY_LABELS.get(category, category.replace("_", " ").title())

        findings.append(
            _make_suggestion(
                page_index=page_index,
                category=category,
                confidence=confidence,
                rects=rects,
                raw_preview=raw_preview,
                reason=model_finding.reason.strip() or "The vision model identified this region as potentially sensitive.",
                source=f"Fireworks vision ({model_name})",
                token_ids=[token.token_id for token in unique_tokens],
            )
        )

    return findings, [], True

def _rects_overlap(first: SensitiveSuggestion, second: SensitiveSuggestion) -> bool:
    return any(_rect_intersection_ratio(a, b) >= 0.45 for a in first.rects for b in second.rects)


def _merge_suggestions(suggestions: Sequence[SensitiveSuggestion]) -> list[SensitiveSuggestion]:
    merged: list[SensitiveSuggestion] = []
    for candidate in sorted(suggestions, key=lambda item: item.confidence, reverse=True):
        match: SensitiveSuggestion | None = None
        for existing in merged:
            same_tokens = bool(set(candidate.token_ids) & set(existing.token_ids))
            same_region = _rects_overlap(candidate, existing)
            compatible_category = candidate.category == existing.category or {
                candidate.category,
                existing.category,
            } <= {"api_key", "access_token", "password"}
            if candidate.page_index == existing.page_index and compatible_category and (same_tokens or same_region):
                match = existing
                break
        if match is None:
            merged.append(candidate)
            continue

        combined_rects = list(match.rects)
        for rect in candidate.rects:
            if not any(_rect_intersection_ratio(rect, existing_rect) >= 0.88 for existing_rect in combined_rects):
                combined_rects.append(rect)
        match.rects = combined_rects
        match.token_ids = tuple(dict.fromkeys((*match.token_ids, *candidate.token_ids)))
        match.confidence = min(0.995, 1 - (1 - match.confidence) * (1 - candidate.confidence))
        if candidate.source not in match.source:
            match.source = f"{match.source} + {candidate.source}"
        if len(candidate.reason) > len(match.reason):
            match.reason = candidate.reason
        match.suggestion_id = _suggestion_id(match.page_index, match.category, match.rects, match.preview)

    return sorted(merged, key=lambda item: (-item.confidence, item.page_index, item.category))


def _pad_rect(rect: Rect, amount: float, view: PageView) -> Rect:
    x0, y0, x1, y1 = normalise_rect(rect)
    return (
        max(0.0, x0 - amount),
        max(0.0, y0 - amount),
        min(view.page_width, x1 + amount),
        min(view.page_height, y1 + amount),
    )


def analyze_page(
    *,
    page_index: int,
    image: Image.Image,
    view: PageView,
    embedded_words: Sequence[WordBox] = (),
    use_ai: bool = True,
    run_ocr: bool = True,
    custom_instruction: str = "",
) -> AnalysisResult:
    embedded = embedded_tokens(embedded_words, page_index)
    warnings: list[str] = []

    ocr: list[AnalysisToken] = []
    if run_ocr:
        try:
            ocr = ocr_tokens(image, view, page_index)
        except Exception as exc:
            warnings.append(f"Local OCR failed on page {page_index + 1}: {exc}")

    tokens = merge_embedded_and_ocr(embedded, ocr)
    suggestions = local_pattern_suggestions(tokens, page_index)

    try:
        suggestions.extend(detect_qr_codes(image, view, page_index))
    except Exception as exc:
        warnings.append(f"QR detection failed on page {page_index + 1}: {exc}")

    suggestions.extend(
        custom_prompt_local_suggestions(
            custom_instruction=custom_instruction,
            tokens=tokens,
            image=image,
            view=view,
            page_index=page_index,
        )
    )

    ai_used = False
    if use_ai:
        ai_findings, ai_warnings, ai_used = _call_fireworks_vision(
            image=image,
            view=view,
            tokens=tokens,
            page_index=page_index,
            custom_instruction=custom_instruction,
        )
        suggestions.extend(ai_findings)
        warnings.extend(ai_warnings)

    # Chat findings are expanded into one panel-like region so the review box can cover a full bubble/message area.
    for suggestion in suggestions:
        if suggestion.category == "private_chat" and suggestion.rects:
            x0 = min(rect[0] for rect in suggestion.rects)
            y0 = min(rect[1] for rect in suggestion.rects)
            x1 = max(rect[2] for rect in suggestion.rects)
            y1 = max(rect[3] for rect in suggestion.rects)
            suggestion.rects = [_pad_rect((x0, y0, x1, y1), 6.0, view)]

    merged = _merge_suggestions(suggestions)
    apply_local_calibration(merged)
    return AnalysisResult(
        suggestions=merged,
        warnings=tuple(dict.fromkeys(warnings)),
        token_count=len(tokens),
        ai_used=ai_used,
    )
