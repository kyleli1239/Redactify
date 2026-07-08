from __future__ import annotations

from dataclasses import dataclass, field
import os

from nicegui import events, run, ui

from ai_service import SensitiveSuggestion, analyze_page
from pdf_service import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    PageView,
    PdfError,
    RedactionResult,
    WordBox,
    build_redacted_image,
    build_redacted_pdf,
    detect_upload_kind,
    extract_page_words,
    normalise_rect,
    render_image,
    render_page,
    validate_image,
    validate_pdf,
)

MIN_RECT_SIZE = 2.0
CLICK_TOLERANCE = 3.0
RECT_MATCH_TOLERANCE = 0.8
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@dataclass
class EditorState:
    file_bytes: bytes | None = None
    filename: str = "document.pdf"
    kind: str | None = None  # "pdf" or "image"
    page_count: int = 0
    current_page: int = 0
    page_view: PageView | None = None
    redactions: dict[int, list[tuple[float, float, float, float]]] = field(default_factory=dict)
    words_by_page: dict[int, list[WordBox]] = field(default_factory=dict)
    drawing: bool = False
    start_point: tuple[float, float] | None = None
    draft_rect: tuple[float, float, float, float] | None = None
    final_result: RedactionResult | None = None
    ai_suggestions: list[SensitiveSuggestion] = field(default_factory=list)


@ui.page("/")
def main_page() -> None:
    state = EditorState()

    ui.add_css(
        """
        body { background: #f5f7fa; }
        .app-shell { max-width: 1500px; margin: 0 auto; }
        .editor-image { width: min(100%, 820px); border-radius: 10px; overflow: hidden; }
        .preview-image { width: min(100%, 520px); border-radius: 10px; overflow: hidden; }
        .muted { color: #64748b; }
        """
    )

    with ui.column().classes("app-shell w-full gap-4 p-4"):
        ui.label("Document & Image Redactor").classes("text-3xl font-bold")
        ui.label(
            "Upload a PDF or image, select embedded PDF text or draw boxes, preview the result, then download a permanently redacted copy."
        ).classes("muted")

        with ui.card().classes("w-full"):
            ui.label("1. Upload").classes("text-xl font-semibold")
            upload_status = ui.label("No file uploaded.").classes("muted")

            async def handle_upload(event: events.UploadEventArguments) -> None:
                try:
                    if event.file.size() > MAX_UPLOAD_BYTES:
                        raise PdfError("The file is larger than the 50 MB starter limit.")

                    file_bytes = await event.file.read()
                    kind = await run.io_bound(detect_upload_kind, file_bytes)

                    if kind == "pdf":
                        page_count, _ = await run.io_bound(validate_pdf, file_bytes)
                        detail = f"{page_count} page(s)"
                    else:
                        width, height, image_format = await run.io_bound(validate_image, file_bytes)
                        page_count = 1
                        detail = f"{image_format} image — {width} × {height}px"
                except PdfError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                except Exception as exc:
                    ui.notify(f"Upload failed: {exc}", type="negative")
                    return

                state.file_bytes = file_bytes
                state.filename = event.file.name
                state.kind = kind
                state.page_count = page_count
                state.current_page = 0
                state.redactions = {index: [] for index in range(page_count)}
                state.words_by_page.clear()
                state.final_result = None
                state.ai_suggestions.clear()

                page_select.set_options(list(range(1, page_count + 1)), value=1)
                if kind == "pdf":
                    selection_mode.set_options(
                        {"box": "Draw boxes", "text": "Select embedded PDF text"},
                        value="box",
                    )
                    scrub_checkbox.visible = True
                    scrub_explanation.visible = True
                else:
                    selection_mode.set_options({"box": "Draw boxes"}, value="box")
                    scrub_checkbox.visible = False
                    scrub_explanation.visible = False

                upload_status.set_text(f"{event.file.name} — {detail}")
                editor_controls.visible = True
                workspace.visible = True
                await show_page(0)
                render_suggestions()
                invalidate_final_preview()
                ui.notify(
                    "File loaded. Choose a selection mode and mark the content to remove.",
                    type="positive",
                )

            ui.upload(
                label="Choose PDF or image",
                on_upload=handle_upload,
                auto_upload=True,
                max_file_size=MAX_UPLOAD_BYTES,
            ).props('accept="application/pdf,.pdf,image/jpeg,.jpg,.jpeg,image/png,.png,image/webp,.webp,image/bmp,.bmp"').classes(
                "w-full"
            )

        editor_controls = ui.row().classes("w-full items-center gap-2 flex-wrap")
        editor_controls.visible = False

        workspace = ui.row().classes("w-full items-start gap-4 flex-wrap")
        workspace.visible = False

        with editor_controls:
            previous_button = ui.button("Previous", icon="chevron_left")
            page_select = ui.select(options=[1], value=1, label="Page").classes("w-28")
            next_button = ui.button("Next", icon="chevron_right")
            ui.separator().props("vertical")
            selection_mode = ui.select(
                options={"box": "Draw boxes", "text": "Select embedded PDF text"},
                value="box",
                label="Selection mode",
            ).classes("w-64")
            undo_button = ui.button("Undo", icon="undo")
            clear_button = ui.button("Clear page", icon="delete_outline")
            rectangle_count = ui.label("0 redactions on this page").classes("muted")

        with workspace:
            with ui.card().classes("grow min-w-[680px]"):
                ui.label("2. Mark redactions").classes("text-xl font-semibold")
                mode_help = ui.label(
                    "Draw boxes: click and drag over any area you want to remove."
                ).classes("muted")
                editor_image = ui.interactive_image(
                    source="",
                    size=(CANVAS_WIDTH, CANVAS_HEIGHT),
                    events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                    cross=True,
                ).classes("editor-image")

            with ui.column().classes("w-[560px] max-w-full gap-4"):
                with ui.card().classes("w-full"):
                    ui.label("3. AI suggestions").classes("text-xl font-semibold")
                    ui.label(
                        "The scan proposes likely sensitive regions with confidence scores. Nothing is redacted until you select suggestions and apply them."
                    ).classes("muted text-sm")
                    ui.label(
                        "The rendered page image and extracted text are sent to a Fireworks vision model. Pattern, OCR and QR checks also run locally."
                    ).classes("muted text-xs")

                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        analysis_scope = ui.select(
                            options={"all": "All pages", "current": "Current page"},
                            value="all",
                            label="Scan scope",
                        ).classes("w-40")
                        run_ocr_checkbox = ui.checkbox("OCR scans and images", value=True)

                    threshold_label = ui.label("Show suggestions at 60% confidence or higher").classes("muted text-sm")
                    confidence_threshold = ui.slider(min=0.5, max=0.99, step=0.01, value=0.60).classes("w-full")
                    analyze_button = ui.button("Scan for sensitive information", icon="auto_awesome").props("color=primary")
                    analysis_status = ui.label("Upload a file, then run the scan.").classes("muted")

                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        select_all_button = ui.button("Select all", icon="done_all").props("outline")
                        clear_selection_button = ui.button("Clear selection", icon="remove_done").props("outline")
                        apply_suggestions_button = ui.button("Apply selected", icon="playlist_add_check").props("color=warning")

                    suggestions_container = ui.column().classes("w-full gap-2 max-h-[540px] overflow-auto")

                with ui.card().classes("w-full"):
                    ui.label("4. Final preview").classes("text-xl font-semibold")
                    scrub_checkbox = ui.checkbox(
                        "Remove metadata, hidden text, attachments and JavaScript",
                        value=True,
                    )
                    scrub_explanation = ui.label(
                        "This extra sanitisation applies to PDFs. Image exports are rewritten without the original metadata."
                    ).classes("muted text-sm")
                    ui.label(
                        "The preview is regenerated from the final output file. It is not merely a removable overlay."
                    ).classes("muted text-sm")
                    generate_button = ui.button("Generate final preview", icon="visibility").props("color=primary")
                    preview_status = ui.label("Generate a preview after adding redactions.").classes("muted")
                    preview_image = ui.image().classes("preview-image")
                    preview_image.visible = False
                    download_button = ui.button("Download redacted file", icon="download").props("color=positive")
                    download_button.disable()

        def current_rectangles() -> list[tuple[float, float, float, float]]:
            return state.redactions.setdefault(state.current_page, [])

        def current_words() -> list[WordBox]:
            return state.words_by_page.get(state.current_page, [])

        def update_count() -> None:
            count = len(current_rectangles()) if state.file_bytes else 0
            noun = "redaction" if count == 1 else "redactions"
            rectangle_count.set_text(f"{count} {noun} on this page")

        def update_mode_help() -> None:
            if state.kind == "image":
                mode_help.set_text(
                    "Images do not contain embedded selectable text. Click and drag boxes over the pixels to remove."
                )
                return

            if selection_mode.value == "text":
                word_count = len(current_words())
                if word_count:
                    mode_help.set_text(
                        f"Text mode: click a word to toggle it, or drag across text to select multiple words. "
                        f"{word_count} embedded word(s) detected on this page."
                    )
                else:
                    mode_help.set_text(
                        "No embedded text was detected on this page. It may be scanned; use Draw boxes or run the AI/OCR scan."
                    )
            else:
                mode_help.set_text(
                    "Draw boxes: click and drag over any area you want to remove, including graphics or scanned text."
                )

        def overlay_svg() -> str:
            if state.page_view is None:
                return ""

            shapes: list[str] = []
            for x0, y0, x1, y1 in current_rectangles():
                left, top = state.page_view.page_to_canvas(min(x0, x1), min(y0, y1))
                right, bottom = state.page_view.page_to_canvas(max(x0, x1), max(y0, y1))
                shapes.append(
                    f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" '
                    'fill="black" fill-opacity="0.64" stroke="#ef4444" stroke-width="2" />'
                )

            threshold = float(confidence_threshold.value or 0.60)
            for suggestion in state.ai_suggestions:
                if suggestion.page_index != state.current_page or suggestion.applied or suggestion.confidence < threshold:
                    continue
                selected = suggestion.selected
                stroke = "#f59e0b" if selected else "#94a3b8"
                opacity = "0.20" if selected else "0.06"
                for x0, y0, x1, y1 in suggestion.rects:
                    left, top = state.page_view.page_to_canvas(min(x0, x1), min(y0, y1))
                    right, bottom = state.page_view.page_to_canvas(max(x0, x1), max(y0, y1))
                    shapes.append(
                        f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" '
                        f'fill="#f59e0b" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="2" stroke-dasharray="6 4" />'
                    )

            if state.draft_rect is not None:
                x0, y0, x1, y1 = state.draft_rect
                left, top = state.page_view.page_to_canvas(min(x0, x1), min(y0, y1))
                right, bottom = state.page_view.page_to_canvas(max(x0, x1), max(y0, y1))
                if selection_mode.value == "text":
                    fill = "#2563eb"
                    opacity = "0.18"
                else:
                    fill = "black"
                    opacity = "0.38"
                shapes.append(
                    f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" '
                    f'fill="{fill}" fill-opacity="{opacity}" stroke="#2563eb" stroke-width="2" stroke-dasharray="8 5" />'
                )

            return "".join(shapes)

        def refresh_overlay() -> None:
            editor_image.set_content(overlay_svg())
            update_count()
            update_mode_help()

        def invalidate_final_preview() -> None:
            state.final_result = None
            preview_status.set_text("Changes made. Generate a new final preview.")
            preview_image.visible = False
            download_button.disable()

        def visible_suggestions() -> list[SensitiveSuggestion]:
            threshold = float(confidence_threshold.value or 0.60)
            return [
                suggestion
                for suggestion in state.ai_suggestions
                if suggestion.confidence >= threshold
            ]

        def render_suggestions() -> None:
            suggestions_container.clear()
            suggestions = visible_suggestions()
            with suggestions_container:
                if not suggestions:
                    ui.label("No suggestions at the current confidence threshold.").classes("muted text-sm")
                    return

                for suggestion in suggestions:
                    with ui.card().classes("w-full p-3"):
                        with ui.row().classes("w-full items-start gap-2"):
                            checkbox = ui.checkbox(value=suggestion.selected and not suggestion.applied)
                            checkbox.set_enabled(not suggestion.applied)

                            def update_selection(event: events.ValueChangeEventArguments, item=suggestion) -> None:
                                item.selected = bool(event.value)
                                refresh_overlay()

                            checkbox.on_value_change(update_selection)
                            with ui.column().classes("grow gap-0"):
                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                    ui.badge(suggestion.category_label)
                                    ui.label(f"{suggestion.confidence * 100:.0f}% • Page {suggestion.page_index + 1}").classes(
                                        "font-semibold"
                                    )
                                    if suggestion.applied:
                                        ui.badge("Applied").props("color=positive")
                                ui.label(suggestion.preview).classes("text-sm break-all")
                                ui.label(f"{suggestion.reason} Source: {suggestion.source}.").classes(
                                    "muted text-xs"
                                )
                                if suggestion.page_index != state.current_page:
                                    ui.button(
                                        "Show page",
                                        icon="find_in_page",
                                        on_click=lambda page=suggestion.page_index: show_page(page),
                                    ).props("flat dense").classes("self-start")

        def threshold_changed(event: events.ValueChangeEventArguments) -> None:
            value = float(event.value or 0.60)
            threshold_label.set_text(f"Show suggestions at {value * 100:.0f}% confidence or higher")
            render_suggestions()
            refresh_overlay()

        confidence_threshold.on_value_change(threshold_changed)

        async def show_page(page_index: int) -> None:
            if state.file_bytes is None or state.kind is None:
                return

            page_index = min(max(page_index, 0), state.page_count - 1)
            state.current_page = page_index
            state.drawing = False
            state.start_point = None
            state.draft_rect = None

            if state.kind == "pdf":
                image, view = await run.io_bound(render_page, state.file_bytes, page_index)
                if page_index not in state.words_by_page:
                    state.words_by_page[page_index] = await run.io_bound(
                        extract_page_words, state.file_bytes, page_index
                    )
            else:
                image, view = await run.io_bound(render_image, state.file_bytes)
                state.words_by_page[page_index] = []

            state.page_view = view
            editor_image.set_source(image)
            page_select.set_value(page_index + 1)
            refresh_overlay()

            previous_button.set_enabled(page_index > 0)
            next_button.set_enabled(page_index < state.page_count - 1)

            if state.final_result is not None:
                if state.kind == "pdf":
                    final_image, _ = await run.io_bound(
                        render_page, state.final_result.output_bytes, page_index
                    )
                else:
                    final_image, _ = await run.io_bound(
                        render_image, state.final_result.output_bytes
                    )
                preview_image.set_source(final_image)
                preview_image.visible = True

        def rectangles_match(
            first: tuple[float, float, float, float],
            second: tuple[float, float, float, float],
        ) -> bool:
            a = normalise_rect(first)
            b = normalise_rect(second)
            return all(abs(left - right) <= RECT_MATCH_TOLERANCE for left, right in zip(a, b))

        def find_existing_rectangle(rect: tuple[float, float, float, float]) -> int | None:
            for index, existing in enumerate(current_rectangles()):
                if rectangles_match(existing, rect):
                    return index
            return None

        def rects_intersect(
            first: tuple[float, float, float, float],
            second: tuple[float, float, float, float],
        ) -> bool:
            ax0, ay0, ax1, ay1 = normalise_rect(first)
            bx0, by0, bx1, by1 = normalise_rect(second)
            return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0

        def word_at_point(x: float, y: float) -> WordBox | None:
            matches = [
                word
                for word in current_words()
                if word.rect[0] <= x <= word.rect[2] and word.rect[1] <= y <= word.rect[3]
            ]
            if not matches:
                return None
            return min(
                matches,
                key=lambda word: (word.rect[2] - word.rect[0]) * (word.rect[3] - word.rect[1]),
            )

        def select_embedded_text(
            start_x: float,
            start_y: float,
            current_x: float,
            current_y: float,
        ) -> int:
            width = abs(current_x - start_x)
            height = abs(current_y - start_y)

            if width <= CLICK_TOLERANCE and height <= CLICK_TOLERANCE:
                word = word_at_point(current_x, current_y)
                if word is None:
                    return 0
                existing_index = find_existing_rectangle(word.rect)
                if existing_index is not None:
                    current_rectangles().pop(existing_index)
                else:
                    current_rectangles().append(word.rect)
                return 1

            selection = normalise_rect((start_x, start_y, current_x, current_y))
            selected = [word for word in current_words() if rects_intersect(selection, word.rect)]
            additions = 0
            for word in selected:
                if find_existing_rectangle(word.rect) is None:
                    current_rectangles().append(word.rect)
                    additions += 1
            return additions

        def mouse_handler(event: events.MouseEventArguments) -> None:
            if state.file_bytes is None or state.page_view is None:
                return

            if event.type == "mousedown":
                if not state.page_view.contains_canvas_point(event.image_x, event.image_y):
                    return
                state.drawing = True
                state.start_point = state.page_view.canvas_to_page(event.image_x, event.image_y)
                x, y = state.start_point
                state.draft_rect = (x, y, x, y)
                refresh_overlay()
                return

            if event.type == "mouseleave":
                state.drawing = False
                state.start_point = None
                state.draft_rect = None
                refresh_overlay()
                return

            if not state.drawing or state.start_point is None:
                return

            current_x, current_y = state.page_view.canvas_to_page(event.image_x, event.image_y)
            start_x, start_y = state.start_point
            state.draft_rect = (start_x, start_y, current_x, current_y)

            if event.type == "mousemove":
                if event.buttons == 0:
                    state.drawing = False
                    state.start_point = None
                    state.draft_rect = None
                refresh_overlay()
                return

            if event.type == "mouseup":
                changed = False
                if selection_mode.value == "text" and state.kind == "pdf":
                    changed = select_embedded_text(start_x, start_y, current_x, current_y) > 0
                    if not changed and not current_words():
                        ui.notify(
                            "No embedded text is available on this page. Use Draw boxes for scanned content.",
                            type="warning",
                        )
                else:
                    width = abs(current_x - start_x)
                    height = abs(current_y - start_y)
                    if width >= MIN_RECT_SIZE and height >= MIN_RECT_SIZE:
                        current_rectangles().append((start_x, start_y, current_x, current_y))
                        changed = True

                if changed:
                    invalidate_final_preview()
                state.drawing = False
                state.start_point = None
                state.draft_rect = None
                refresh_overlay()

        editor_image.on_mouse(mouse_handler)

        async def change_page(event: events.ValueChangeEventArguments) -> None:
            if event.value is None:
                return
            await show_page(int(event.value) - 1)

        def change_selection_mode(_: events.ValueChangeEventArguments) -> None:
            state.drawing = False
            state.start_point = None
            state.draft_rect = None
            refresh_overlay()

        page_select.on_value_change(change_page)
        selection_mode.on_value_change(change_selection_mode)
        previous_button.on_click(lambda: show_page(state.current_page - 1))
        next_button.on_click(lambda: show_page(state.current_page + 1))

        def undo_redaction() -> None:
            if current_rectangles():
                current_rectangles().pop()
                invalidate_final_preview()
                refresh_overlay()

        def clear_page_redactions() -> None:
            if current_rectangles():
                current_rectangles().clear()
                invalidate_final_preview()
                refresh_overlay()

        undo_button.on_click(undo_redaction)
        clear_button.on_click(clear_page_redactions)

        async def scan_for_sensitive_information() -> None:
            if state.file_bytes is None or state.kind is None:
                ui.notify("Upload a PDF or image first.", type="warning")
                return

            if analysis_scope.value == "current":
                page_indexes = [state.current_page]
                state.ai_suggestions = [
                    suggestion for suggestion in state.ai_suggestions if suggestion.page_index != state.current_page
                ]
            else:
                page_indexes = list(range(state.page_count))
                state.ai_suggestions.clear()

            analyze_button.disable()
            apply_suggestions_button.disable()
            warnings: list[str] = []
            total_tokens = 0
            ai_was_used = False

            try:
                for position, page_index in enumerate(page_indexes, start=1):
                    analysis_status.set_text(
                        f"Scanning page {page_index + 1} ({position}/{len(page_indexes)})…"
                    )
                    if state.kind == "pdf":
                        image, view = await run.io_bound(render_page, state.file_bytes, page_index)
                        words = state.words_by_page.get(page_index)
                        if words is None:
                            words = await run.io_bound(extract_page_words, state.file_bytes, page_index)
                            state.words_by_page[page_index] = words
                    else:
                        image, view = await run.io_bound(render_image, state.file_bytes)
                        words = []

                    result = await run.io_bound(
                        analyze_page,
                        page_index=page_index,
                        image=image,
                        view=view,
                        embedded_words=words,
                        use_ai=True,
                        run_ocr=bool(run_ocr_checkbox.value),
                    )
                    state.ai_suggestions.extend(result.suggestions)
                    warnings.extend(result.warnings)
                    total_tokens += result.token_count
                    ai_was_used = ai_was_used or result.ai_used
                    render_suggestions()
                    refresh_overlay()

                visible_count = len(visible_suggestions())
                status = (
                    f"Found {len(state.ai_suggestions)} suggestion(s); {visible_count} meet the current threshold. "
                    f"Analysed {total_tokens} text token(s)."
                )
                if not ai_was_used:
                    status += " The Fireworks vision model did not run; local detectors may still have produced suggestions."
                analysis_status.set_text(status)
                if warnings:
                    ui.notify(warnings[0], type="warning", timeout=10000)
                else:
                    ui.notify("Sensitive-information scan complete.", type="positive")
            except Exception as exc:
                analysis_status.set_text("The scan could not be completed.")
                ui.notify(f"AI scan failed: {exc}", type="negative")
            finally:
                analyze_button.enable()
                apply_suggestions_button.enable()

        analyze_button.on_click(scan_for_sensitive_information)

        def set_visible_selection(value: bool) -> None:
            for suggestion in visible_suggestions():
                if not suggestion.applied:
                    suggestion.selected = value
            render_suggestions()
            refresh_overlay()

        select_all_button.on_click(lambda: set_visible_selection(True))
        clear_selection_button.on_click(lambda: set_visible_selection(False))

        def apply_selected_suggestions() -> None:
            chosen = [
                suggestion
                for suggestion in visible_suggestions()
                if suggestion.selected and not suggestion.applied
            ]
            if not chosen:
                ui.notify("Select at least one unapplied suggestion.", type="warning")
                return

            added_rectangles = 0
            for suggestion in chosen:
                page_rectangles = state.redactions.setdefault(suggestion.page_index, [])
                for rect in suggestion.rects:
                    if not any(rectangles_match(rect, existing) for existing in page_rectangles):
                        page_rectangles.append(rect)
                        added_rectangles += 1
                suggestion.applied = True
                suggestion.selected = False

            invalidate_final_preview()
            render_suggestions()
            refresh_overlay()
            ui.notify(
                f"Applied {len(chosen)} suggestion(s) as {added_rectangles} permanent-redaction region(s).",
                type="positive",
            )

        apply_suggestions_button.on_click(apply_selected_suggestions)

        async def generate_final_preview() -> None:
            if state.file_bytes is None or state.kind is None:
                return
            total_redactions = sum(len(rectangles) for rectangles in state.redactions.values())
            if total_redactions == 0:
                ui.notify("Add at least one redaction first.", type="warning")
                return

            generate_button.disable()
            preview_status.set_text("Building permanent redactions…")
            try:
                if state.kind == "pdf":
                    result = await run.io_bound(
                        build_redacted_pdf,
                        state.file_bytes,
                        state.redactions,
                        scrub_hidden_content=bool(scrub_checkbox.value),
                        original_name=state.filename,
                    )
                    final_image, _ = await run.io_bound(
                        render_page, result.output_bytes, state.current_page
                    )
                else:
                    result = await run.io_bound(
                        build_redacted_image,
                        state.file_bytes,
                        state.redactions.get(0, []),
                        original_name=state.filename,
                    )
                    final_image, _ = await run.io_bound(render_image, result.output_bytes)

                state.final_result = result
                preview_image.set_source(final_image)
                preview_image.visible = True
                status = f"Ready: {total_redactions} permanent redaction(s)."
                if result.warning:
                    status += f" {result.warning}"
                preview_status.set_text(status)
                download_button.enable()
                if result.warning:
                    ui.notify(result.warning, type="warning", timeout=8000)
                else:
                    ui.notify("Final redacted file generated.", type="positive")
            except Exception as exc:
                state.final_result = None
                preview_image.visible = False
                download_button.disable()
                preview_status.set_text("Could not generate the redacted file.")
                ui.notify(f"Redaction failed: {exc}", type="negative")
            finally:
                generate_button.enable()

        generate_button.on_click(generate_final_preview)

        def download_output() -> None:
            if state.final_result is None:
                ui.notify("Generate the final preview first.", type="warning")
                return
            ui.download(
                state.final_result.output_bytes,
                filename=state.final_result.filename,
                media_type=state.final_result.media_type,
            )

        download_button.on_click(download_output)

        with ui.card().classes("w-full"):
            ui.label("How manual and AI-assisted redaction work").classes("font-semibold")
            ui.label(
                "Select embedded PDF text snaps to real word coordinates and permanently removes those PDF objects. "
                "Draw boxes works on PDFs, scanned pages, graphics and image files. For images, the selected pixels "
                "are overwritten and the file is re-encoded rather than covered with a removable overlay. "
                "AI suggestions use the same real text boxes or pixel regions, but remain optional until applied."
            ).classes("muted text-sm")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Document & Image Redactor",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8081")),
        reload=False,
    )
