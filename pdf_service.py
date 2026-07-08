from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pymupdf
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 1300
CANVAS_PADDING = 24
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class PdfError(ValueError):
    """Raised when an uploaded document or image cannot be processed safely."""


@dataclass(frozen=True)
class RedactionResult:
    output_bytes: bytes
    media_type: str
    filename: str
    warning: str | None = None

    @property
    def pdf_bytes(self) -> bytes:
        """Backward-compatible alias used by the earlier starter."""
        return self.output_bytes


@dataclass(frozen=True)
class WordBox:
    rect: tuple[float, float, float, float]
    text: str


@dataclass(frozen=True)
class PageView:
    """Mapping between fixed canvas pixels and visible document coordinates."""

    scale: float
    offset_x: float
    offset_y: float
    page_width: float
    page_height: float

    @property
    def pixel_width(self) -> float:
        return self.page_width * self.scale

    @property
    def pixel_height(self) -> float:
        return self.page_height * self.scale

    def contains_canvas_point(self, x: float, y: float) -> bool:
        return (
            self.offset_x <= x <= self.offset_x + self.pixel_width
            and self.offset_y <= y <= self.offset_y + self.pixel_height
        )

    def canvas_to_page(self, x: float, y: float) -> tuple[float, float]:
        page_x = (x - self.offset_x) / self.scale
        page_y = (y - self.offset_y) / self.scale
        return self.clamp_page_point(page_x, page_y)

    def page_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def clamp_page_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, 0.0), self.page_width),
            min(max(y, 0.0), self.page_height),
        )


def _build_view(page_width: float, page_height: float) -> PageView:
    if page_width <= 0 or page_height <= 0:
        raise PdfError("The uploaded file has invalid dimensions.")

    usable_width = CANVAS_WIDTH - 2 * CANVAS_PADDING
    usable_height = CANVAS_HEIGHT - 2 * CANVAS_PADDING
    scale = min(usable_width / page_width, usable_height / page_height)
    pixel_width = page_width * scale
    pixel_height = page_height * scale

    return PageView(
        scale=scale,
        offset_x=(CANVAS_WIDTH - pixel_width) / 2,
        offset_y=(CANVAS_HEIGHT - pixel_height) / 2,
        page_width=page_width,
        page_height=page_height,
    )


def validate_pdf(pdf_bytes: bytes) -> tuple[int, str]:
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise PdfError("The selected file does not appear to be a PDF.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise PdfError("Password-protected PDFs are not supported in this starter.")
            if document.page_count < 1:
                raise PdfError("The PDF has no pages.")
            return document.page_count, document.metadata.get("title") or ""
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(f"Could not open the PDF: {exc}") from exc


def _open_flat_image(image_bytes: bytes) -> tuple[Image.Image, str]:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image_format = (source.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise PdfError(f"Unsupported image type. Use one of: {supported}.")

            oriented = ImageOps.exif_transpose(source)
            oriented.load()
            width, height = oriented.size
            if width < 1 or height < 1:
                raise PdfError("The image has invalid dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise PdfError(
                    f"The image is too large. The current limit is {MAX_IMAGE_PIXELS:,} pixels."
                )

            if oriented.mode in {"RGBA", "LA"} or (
                oriented.mode == "P" and "transparency" in oriented.info
            ):
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                flattened = background.convert("RGB")
            else:
                flattened = oriented.convert("RGB")

            return flattened.copy(), image_format
    except PdfError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise PdfError("The selected file is not a readable supported image.") from exc


def validate_image(image_bytes: bytes) -> tuple[int, int, str]:
    image, image_format = _open_flat_image(image_bytes)
    return image.width, image.height, image_format


def detect_upload_kind(file_bytes: bytes) -> str:
    if b"%PDF-" in file_bytes[:1024]:
        validate_pdf(file_bytes)
        return "pdf"
    validate_image(file_bytes)
    return "image"


def get_page_view(pdf_bytes: bytes, page_index: int) -> PageView:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document[page_index]
        return _build_view(float(page.rect.width), float(page.rect.height))


def render_page(pdf_bytes: bytes, page_index: int) -> tuple[Image.Image, PageView]:
    """Render one PDF page onto a fixed-size canvas."""

    view = get_page_view(pdf_bytes, page_index)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document[page_index]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(view.scale, view.scale), alpha=False)

    page_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#dfe3e8")
    canvas.paste(page_image, (round(view.offset_x), round(view.offset_y)))
    return canvas, view


def render_image(image_bytes: bytes) -> tuple[Image.Image, PageView]:
    """Render an uploaded image onto the same fixed editor canvas."""

    image, _ = _open_flat_image(image_bytes)
    view = _build_view(float(image.width), float(image.height))
    rendered_width = max(1, round(view.pixel_width))
    rendered_height = max(1, round(view.pixel_height))
    rendered = image.resize((rendered_width, rendered_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#dfe3e8")
    canvas.paste(rendered, (round(view.offset_x), round(view.offset_y)))
    return canvas, view


def extract_page_words(pdf_bytes: bytes, page_index: int) -> list[WordBox]:
    """Return selectable embedded PDF words in visible, rotated-page coordinates."""

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document[page_index]
        rotation_matrix = page.rotation_matrix
        page_rect = page.rect
        raw_words = page.get_text("words", sort=True)

        words: list[WordBox] = []
        for item in raw_words:
            if len(item) < 5:
                continue
            text = str(item[4]).strip()
            if not text:
                continue

            unrotated_rect = pymupdf.Rect(float(item[0]), float(item[1]), float(item[2]), float(item[3]))
            visible_rect = unrotated_rect * rotation_matrix
            visible_rect &= page_rect
            if visible_rect.is_empty or visible_rect.width <= 0 or visible_rect.height <= 0:
                continue

            words.append(
                WordBox(
                    rect=(
                        float(visible_rect.x0),
                        float(visible_rect.y0),
                        float(visible_rect.x1),
                        float(visible_rect.y1),
                    ),
                    text=text,
                )
            )

        return words


def normalise_rect(rect: Iterable[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in rect)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _is_missing_xref_error(exc: Exception) -> bool:
    return "cannot find object in xref" in str(exc).lower()


def _safe_tobytes(document: pymupdf.Document) -> bytes:
    """Save a PDF with progressively less aggressive cleanup if its xref is damaged."""

    attempts = (
        dict(garbage=3, deflate=True, clean=False),
        dict(garbage=1, deflate=True, clean=False),
        dict(garbage=0, deflate=True, clean=False),
    )
    last_error: Exception | None = None

    for options in attempts:
        try:
            return document.tobytes(**options)
        except RuntimeError as exc:
            last_error = exc
            if not _is_missing_xref_error(exc):
                raise

    raise PdfError(f"The PDF contains a broken cross-reference that could not be repaired: {last_error}")


def _normalise_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """Ask MuPDF to repair malformed xrefs before editing the document."""

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        repair = getattr(document, "repair", None)
        if callable(repair):
            try:
                repair()
            except Exception:
                pass
        return _safe_tobytes(document)


def build_redacted_pdf(
    pdf_bytes: bytes,
    redactions: dict[int, list[tuple[float, float, float, float]]],
    *,
    scrub_hidden_content: bool = True,
    original_name: str = "document.pdf",
) -> RedactionResult:
    """Permanently remove PDF content covered by the supplied visible rectangles."""

    normalised_bytes = _normalise_pdf_bytes(pdf_bytes)

    with pymupdf.open(stream=normalised_bytes, filetype="pdf") as document:
        for page_index, page_rectangles in redactions.items():
            if not page_rectangles:
                continue

            page = document[page_index]
            for raw_rect in page_rectangles:
                visible_rect = pymupdf.Rect(*normalise_rect(raw_rect))
                visible_rect &= page.rect
                if visible_rect.is_empty:
                    continue
                pdf_rect = visible_rect * page.derotation_matrix
                page.add_redact_annot(pdf_rect, fill=(0, 0, 0), cross_out=False)

            page.apply_redactions()

        if scrub_hidden_content:
            try:
                document.set_metadata({})
            except Exception:
                pass
            try:
                document.del_xml_metadata()
            except Exception:
                pass

        redacted_bytes = _safe_tobytes(document)

    filename = output_pdf_filename(original_name)
    if not scrub_hidden_content:
        return RedactionResult(redacted_bytes, "application/pdf", filename)

    try:
        with pymupdf.open(stream=redacted_bytes, filetype="pdf") as document:
            document.scrub(
                attached_files=True,
                clean_pages=True,
                embedded_files=True,
                hidden_text=True,
                javascript=True,
                metadata=True,
                redactions=False,
                redact_images=0,
                remove_links=True,
                reset_fields=True,
                reset_responses=True,
                thumbnails=True,
                xml_metadata=True,
            )
            return RedactionResult(
                _safe_tobytes(document),
                "application/pdf",
                filename,
            )
    except RuntimeError as exc:
        if not _is_missing_xref_error(exc):
            raise
        return RedactionResult(
            redacted_bytes,
            "application/pdf",
            filename,
            "Permanent redactions were applied, but optional deep sanitisation "
            "was skipped because this PDF contains a broken cross-reference.",
        )


def build_redacted_image(
    image_bytes: bytes,
    redactions: list[tuple[float, float, float, float]],
    *,
    original_name: str = "image.png",
) -> RedactionResult:
    """Permanently replace selected image pixels and rewrite the image file."""

    image, source_format = _open_flat_image(image_bytes)
    draw = ImageDraw.Draw(image)

    for raw_rect in redactions:
        x0, y0, x1, y1 = normalise_rect(raw_rect)
        x0 = min(max(x0, 0.0), float(image.width))
        x1 = min(max(x1, 0.0), float(image.width))
        y0 = min(max(y0, 0.0), float(image.height))
        y1 = min(max(y1, 0.0), float(image.height))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((round(x0), round(y0), round(x1), round(y1)), fill="black")

    output = BytesIO()
    if source_format == "JPEG":
        image.save(output, format="JPEG", quality=95, optimize=True)
        media_type = "image/jpeg"
        extension = ".jpg"
    elif source_format == "WEBP":
        image.save(output, format="WEBP", lossless=True, method=6)
        media_type = "image/webp"
        extension = ".webp"
    else:
        image.save(output, format="PNG", optimize=True)
        media_type = "image/png"
        extension = ".png"

    return RedactionResult(
        output.getvalue(),
        media_type,
        output_image_filename(original_name, extension),
    )


def output_pdf_filename(original_name: str) -> str:
    safe_name = Path(original_name).name
    stem = Path(safe_name).stem or "document"
    return f"{stem}_redacted.pdf"


def output_image_filename(original_name: str, extension: str = ".png") -> str:
    safe_name = Path(original_name).name
    stem = Path(safe_name).stem or "image"
    return f"{stem}_redacted{extension}"


def output_filename(original_name: str) -> str:
    """Backward-compatible PDF filename helper."""
    return output_pdf_filename(original_name)
