from __future__ import annotations

from dataclasses import dataclass, field
import copy
import os

from nicegui import events, run, ui

from ai_service import SensitiveSuggestion, analyze_page, validate_fireworks_connection
from feedback_store import record_review_metadata
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

FIREWORKS_MODEL_CATALOG: dict[str, dict[str, str]] = {
    "accounts/fireworks/models/minimax-m3": {
        "label": "MiniMax M3 — Medium · best value",
        "strength": "MEDIUM",
        "billing": "Serverless: $0.30 input / $0.06 cached input / $1.20 output per 1M tokens",
        "description": "Lowest-cost default. Native multimodal analysis with good document and screenshot coverage.",
    },
    "accounts/fireworks/models/qwen3p7-plus": {
        "label": "Qwen 3.7 Plus — Strong · recommended",
        "strength": "STRONG",
        "billing": "Serverless: $0.40 input / $0.08 cached input / $1.60 output per 1M tokens",
        "description": "Best quality-to-cost balance for visual document reasoning, complex layouts and contextual PII.",
    },
    "accounts/fireworks/models/kimi-k2p6": {
        "label": "Kimi K2.6 — Strong · premium",
        "strength": "STRONG",
        "billing": "Serverless: $0.95 input / $0.16 cached input / $4.00 output per 1M tokens",
        "description": "Premium multimodal reasoning for difficult pages and ambiguous visual context; costly for routine scans.",
    },
}
DEFAULT_UI_FIREWORKS_MODEL = "accounts/fireworks/models/minimax-m3"


@dataclass
class HistorySnapshot:
    redactions: dict[int, list[tuple[float, float, float, float]]]
    suggestion_states: dict[str, tuple[bool, bool]]
    current_page: int
    label: str = "Edit"


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
    undo_stack: list[HistorySnapshot] = field(default_factory=list)
    redo_stack: list[HistorySnapshot] = field(default_factory=list)
    fireworks_connected: bool = False
    connected_api_key: str | None = None
    connected_model: str | None = None
    scan_in_progress: bool = False


@ui.page("/")
def main_page() -> None:
    state = EditorState()

    ui.dark_mode(True)
    ui.colors(
        primary="#5eead4",
        secondary="#7dd3fc",
        accent="#a78bfa",
        positive="#6ee7b7",
        negative="#fb7185",
        warning="#fbbf24",
        dark="#07151d",
    )
    ui.add_css(
        """
        :root {
            --ink: #e8fbff;
            --muted: #8faab5;
            --panel: rgba(9, 28, 37, 0.82);
            --panel-strong: rgba(8, 23, 31, 0.96);
            --line: rgba(125, 211, 252, 0.17);
            --aurora: #5eead4;
            --aurora-blue: #7dd3fc;
            --aurora-violet: #a78bfa;
        }
        html, body { min-height: 100%; }
        body {
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 5%, rgba(94,234,212,.15), transparent 30%),
                radial-gradient(circle at 88% 12%, rgba(125,211,252,.14), transparent 32%),
                radial-gradient(circle at 58% 90%, rgba(167,139,250,.10), transparent 34%),
                linear-gradient(145deg, #040c12 0%, #07151d 46%, #061219 100%);
            background-attachment: fixed;
        }
        body, .q-field, .q-btn, .q-checkbox, .q-badge {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        h1, h2, h3, .cyber-title, .section-kicker, .shortcut-chip {
            font-family: "Cascadia Code", "JetBrains Mono", "SFMono-Regular", "Courier New", monospace;
        }
        .app-shell { max-width: 1780px; margin: 0 auto; }
        .hero {
            position: relative; overflow: hidden; border: 1px solid rgba(94,234,212,.22);
            border-radius: 26px; padding: 1.35rem 1.5rem;
            background: linear-gradient(125deg, rgba(8,31,39,.95), rgba(8,24,34,.76));
            box-shadow: 0 28px 80px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.04);
        }
        .hero::after {
            content: ""; position: absolute; inset: -80% -15% auto 45%; height: 260px;
            background: linear-gradient(100deg, transparent, rgba(94,234,212,.18), rgba(125,211,252,.14), transparent);
            transform: rotate(-12deg); filter: blur(18px); pointer-events: none;
        }
        .cyber-title { letter-spacing: -.045em; text-shadow: 0 0 28px rgba(94,234,212,.13); }
        .section-kicker { color: var(--aurora); letter-spacing: .12em; text-transform: uppercase; font-size: .72rem; }
        .q-card {
            color: var(--ink); background: var(--panel); backdrop-filter: blur(18px);
            border: 1px solid var(--line); border-radius: 22px;
            box-shadow: 0 20px 55px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.025);
        }
        .glass-card { transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease; }
        .glass-card:hover { border-color: rgba(94,234,212,.27); box-shadow: 0 22px 65px rgba(0,0,0,.28); }
        .workspace-grid {
            display: grid; grid-template-columns: minmax(0, 1fr) 450px;
            align-items: start; gap: 1.15rem;
        }
        .editor-column { min-width: 0; }
        .toolbar {
            padding: .7rem .8rem; border: 1px solid var(--line); border-radius: 18px;
            background: rgba(6,22,30,.78); backdrop-filter: blur(16px);
            box-shadow: 0 14px 40px rgba(0,0,0,.18);
        }
        .ai-sidebar {
            position: sticky; top: 1rem; height: calc(100vh - 2rem); min-height: 620px;
            overflow-y: auto; overflow-x: hidden;
            border: 1px solid rgba(94,234,212,.26);
            box-shadow: 0 25px 80px rgba(0,0,0,.32), 0 0 45px rgba(94,234,212,.055);
            scrollbar-color: rgba(94,234,212,.48) rgba(3,17,24,.35);
            scrollbar-width: thin;
        }
        .ai-sidebar::-webkit-scrollbar { width: 9px; }
        .ai-sidebar::-webkit-scrollbar-track { background: rgba(3,17,24,.35); border-radius: 10px; }
        .ai-sidebar::-webkit-scrollbar-thumb { background: linear-gradient(#5eead4, #7dd3fc); border-radius: 10px; }
        .ai-sidebar::before {
            content: ""; display: block; height: 3px; margin: -16px -16px 12px;
            background: linear-gradient(90deg, #5eead4, #7dd3fc, #a78bfa);
        }
        .suggestion-scroll {
            min-height: 190px; overflow: visible; padding-right: .15rem;
        }
        .sidebar-section {
            border: 1px solid rgba(125,211,252,.14); border-radius: 18px;
            background: linear-gradient(135deg, rgba(4,22,29,.72), rgba(8,28,38,.48));
            padding: .9rem;
        }
        .connection-card {
            border-color: rgba(94,234,212,.30);
            box-shadow: 0 18px 55px rgba(0,0,0,.20), 0 0 35px rgba(94,234,212,.045);
        }
        .connection-status {
            border: 1px solid rgba(125,211,252,.18); border-radius: 12px;
            padding: .5rem .7rem; background: rgba(3,17,24,.55); line-height: 1.35;
        }
        .progress-shell {
            border: 1px solid rgba(94,234,212,.16); border-radius: 14px;
            padding: .65rem; background: rgba(3,17,24,.48);
        }
        .scan-status { white-space: normal; overflow-wrap: anywhere; line-height: 1.45; }
        .suggestion-card {
            background: rgba(7,24,32,.88); border-radius: 16px;
            border: 1px solid rgba(125,211,252,.13); cursor: pointer;
            user-select: none; transition: border-color .18s ease, background .18s ease, transform .18s ease, box-shadow .18s ease;
        }
        .suggestion-card:hover {
            border-color: rgba(94,234,212,.42); background: rgba(9,34,43,.94);
            transform: translateY(-1px); box-shadow: 0 12px 30px rgba(0,0,0,.18);
        }
        .suggestion-card-selected {
            border-color: rgba(94,234,212,.72) !important;
            background: linear-gradient(135deg, rgba(15,66,68,.62), rgba(10,39,52,.94)) !important;
            box-shadow: 0 0 0 1px rgba(94,234,212,.14), 0 14px 34px rgba(0,0,0,.24);
        }
        .suggestion-card-applied { cursor: default; opacity: .72; }
        .editor-image {
            width: min(100%, 980px); border-radius: 18px; overflow: hidden;
            border: 1px solid rgba(125,211,252,.20); background: #071017;
            box-shadow: 0 22px 60px rgba(0,0,0,.32);
        }
        .preview-image { width: min(100%, 720px); border-radius: 18px; overflow: hidden; border: 1px solid var(--line); }
        .muted { color: var(--muted); }
        .aurora-text {
            background: linear-gradient(90deg, #72f7d6, #85dcff 55%, #b9a4ff);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .shortcut-chip {
            border: 1px solid rgba(125,211,252,.18); border-radius: 10px; padding: .22rem .52rem;
            color: #bdeefa; background: rgba(3,17,24,.70); font-size: .72rem;
        }
        .q-btn { border-radius: 12px; text-transform: none; font-weight: 650; letter-spacing: .005em; }
        .q-field--outlined .q-field__control { border-radius: 14px; background: rgba(3,17,24,.46); }
        .q-field--outlined .q-field__control:before { border-color: rgba(125,211,252,.20); }
        .q-field--outlined.q-field--focused .q-field__control:after { border-color: #5eead4; }
        .q-uploader { border-radius: 16px; overflow: hidden; background: rgba(4,18,25,.62); border: 1px dashed rgba(94,234,212,.30); }
        .q-uploader__header { background: linear-gradient(110deg, rgba(20,91,91,.72), rgba(31,74,105,.62)); }
        .confidence-glow { color: #7dd3fc; }
        .credential-panel, .model-panel {
            border: 1px solid rgba(94,234,212,.17); border-radius: 16px;
            background: linear-gradient(135deg, rgba(4,22,29,.78), rgba(8,28,38,.55));
            padding: .8rem;
        }
        .security-note {
            border-left: 2px solid rgba(94,234,212,.72); padding-left: .65rem;
            color: #9ec2cc; font-size: .72rem; line-height: 1.45;
        }
        .model-strength {
            width: fit-content; border: 1px solid rgba(125,211,252,.22); border-radius: 999px;
            padding: .16rem .55rem; background: rgba(20,65,78,.42); color: #9df8e4;
            font-family: "Cascadia Code", "Courier New", monospace; font-size: .68rem; letter-spacing: .08em;
        }
        @media (max-width: 1180px) {
            .workspace-grid { grid-template-columns: 1fr; }
            .ai-sidebar { position: static; height: auto; min-height: 0; max-height: none; overflow: visible; }
            .suggestion-scroll { min-height: 0; }
        }
        @media (max-width: 700px) {
            .hero { padding: 1.1rem; border-radius: 20px; }
            .workspace-grid { gap: .8rem; }
        }
        """
    )

    with ui.column().classes("app-shell w-full gap-4 p-4 md:p-6"):
        with ui.element("section").classes("hero w-full"):
            with ui.row().classes("w-full items-center justify-between gap-4 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("AURORA // PRIVACY WORKSPACE").classes("section-kicker")
                    ui.label("Document & Image Redactor").classes("cyber-title aurora-text text-3xl md:text-5xl font-black")
                    ui.label(
                        "Human-reviewed AI detection, precise text selection and permanent pixel-level redaction."
                    ).classes("muted max-w-3xl text-sm md:text-base")
                with ui.row().classes("items-center gap-2"):
                    ui.badge("LOCAL REVIEW").props("outline color=positive")
                    ui.icon("verified_user", size="32px").classes("text-teal-300")

        with ui.card().classes("glass-card connection-card w-full p-5"):
            with ui.row().classes("w-full items-start justify-between gap-4 flex-wrap"):
                with ui.column().classes("gap-1 max-w-2xl"):
                    ui.label("01 // FIREWORKS ACCESS").classes("section-kicker")
                    ui.label("Connect your AI provider").classes("text-xl font-bold")
                    ui.label(
                        "Manual redaction works without an API key. Connect Fireworks only when you want AI-powered suggestions."
                    ).classes("muted text-sm")
                connection_status_icon = ui.icon("vpn_key", size="28px").classes("text-slate-400")

            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                with ui.column().classes("credential-panel flex-1 min-w-[300px] gap-2"):
                    ui.label("API CREDENTIAL").classes("section-kicker")
                    fireworks_api_key = ui.input(
                        label="Fireworks API key",
                        placeholder="fw_…",
                        password=True,
                        password_toggle_button=True,
                    ).props("outlined autocomplete=new-password spellcheck=false").classes("w-full")
                    ui.label(
                        "The key is held only in this page session and is never written to disk. A small validation request is sent when you press Connect."
                    ).classes("security-note")

                with ui.column().classes("model-panel flex-1 min-w-[320px] gap-2"):
                    ui.label("VISION MODEL").classes("section-kicker")
                    vision_model_select = ui.select(
                        options={
                            model_id: details["label"]
                            for model_id, details in FIREWORKS_MODEL_CATALOG.items()
                        },
                        value=DEFAULT_UI_FIREWORKS_MODEL,
                        label="Fireworks model",
                    ).props("outlined options-dense").classes("w-full")
                    selected_model_strength = ui.label().classes("model-strength")
                    selected_model_description = ui.label().classes("muted text-xs")
                    selected_model_billing = ui.label().classes("text-xs text-cyan-200")
                    ui.label(
                        "Images are billed as input tokens. Rates are indicative and may change."
                    ).classes("muted text-xs")

            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                connect_button = ui.button("Connect API key", icon="link").props("color=primary unelevated")
                connection_spinner = ui.spinner(size="22px").classes("text-teal-300")
                connection_spinner.visible = False
                connection_status = ui.label(
                    "Not connected — manual redaction is available; connect to enable AI scanning."
                ).classes("connection-status muted text-sm flex-1")

            def update_selected_model_details(*_: object) -> None:
                details = FIREWORKS_MODEL_CATALOG.get(
                    str(vision_model_select.value),
                    FIREWORKS_MODEL_CATALOG[DEFAULT_UI_FIREWORKS_MODEL],
                )
                selected_model_strength.set_text(f"TASK STRENGTH // {details['strength']}")
                selected_model_description.set_text(details["description"])
                selected_model_billing.set_text(details["billing"])

            def invalidate_fireworks_connection(*_: object) -> None:
                if state.scan_in_progress:
                    return
                state.fireworks_connected = False
                state.connected_api_key = None
                state.connected_model = None
                connection_status.set_text(
                    "Connection not validated — press Connect after changing the key or model."
                )
                connection_status.classes(replace="connection-status text-sm text-amber-300 flex-1")
                connection_status_icon.classes(replace="text-amber-300")
                try:
                    analyze_button.disable()
                    progress_label.set_text("Connect a validated Fireworks API key to enable AI scanning.")
                    analysis_status.set_text(
                        "Manual redaction remains available. Connect Fireworks to run an AI privacy scan."
                    )
                except NameError:
                    pass

            async def connect_fireworks() -> None:
                if state.scan_in_progress:
                    return
                api_key = str(fireworks_api_key.value or "").strip()
                model = str(vision_model_select.value or DEFAULT_UI_FIREWORKS_MODEL).strip()
                if not api_key:
                    invalidate_fireworks_connection()
                    connection_status.set_text("Invalid API key — enter a Fireworks key before connecting.")
                    connection_status.classes(replace="connection-status text-sm text-rose-300 flex-1")
                    connection_status_icon.classes(replace="text-rose-300")
                    ui.notify("Enter a Fireworks API key first.", type="warning")
                    return
                if model not in FIREWORKS_MODEL_CATALOG:
                    model = DEFAULT_UI_FIREWORKS_MODEL
                    vision_model_select.set_value(model)

                fireworks_api_key.disable()
                vision_model_select.disable()
                connect_button.disable()
                connection_spinner.visible = True
                connection_status.set_text("Validating the key and selected model with Fireworks…")
                connection_status.classes(replace="connection-status text-sm text-cyan-200 flex-1")
                connection_status_icon.classes(replace="text-cyan-300")
                try:
                    valid, message = await run.io_bound(
                        validate_fireworks_connection, api_key=api_key, model=model
                    )
                    if valid:
                        state.fireworks_connected = True
                        state.connected_api_key = api_key
                        state.connected_model = model
                        connection_status.set_text(message)
                        connection_status.classes(replace="connection-status text-sm text-emerald-300 flex-1")
                        connection_status_icon.classes(replace="text-emerald-300")
                        uploader.enable()
                        analyze_button.set_enabled(state.file_bytes is not None)
                        progress_label.set_text("Ready to scan the uploaded document.")
                        analysis_status.set_text(
                            "Fireworks is connected. Configure the scan and review suggestions before applying them."
                        )
                        ui.notify("API key connected successfully.", type="positive")
                    else:
                        state.fireworks_connected = False
                        state.connected_api_key = None
                        state.connected_model = None
                        connection_status.set_text(message)
                        connection_status.classes(replace="connection-status text-sm text-rose-300 flex-1")
                        connection_status_icon.classes(replace="text-rose-300")
                        uploader.enable()
                        analyze_button.disable()
                        ui.notify("Invalid API key or unavailable model. Manual redaction is still available.", type="negative")
                except Exception as exc:
                    state.fireworks_connected = False
                    state.connected_api_key = None
                    state.connected_model = None
                    connection_status.set_text(f"Connection check failed: {exc}")
                    connection_status.classes(replace="connection-status text-sm text-rose-300 flex-1")
                    connection_status_icon.classes(replace="text-rose-300")
                    uploader.enable()
                    analyze_button.disable()
                    ui.notify("Could not validate Fireworks. Manual redaction is still available.", type="negative")
                finally:
                    connection_spinner.visible = False
                    fireworks_api_key.enable()
                    vision_model_select.enable()
                    connect_button.enable()

            update_selected_model_details()
            vision_model_select.on_value_change(update_selected_model_details)
            vision_model_select.on_value_change(invalidate_fireworks_connection)
            fireworks_api_key.on_value_change(invalidate_fireworks_connection)
            connect_button.on_click(connect_fireworks)

        with ui.card().classes("glass-card w-full p-5"):
            ui.label("02 // INPUT").classes("section-kicker")
            ui.label("Upload a document").classes("text-xl font-bold")
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
                state.undo_stack.clear()
                state.redo_stack.clear()
                update_history_buttons()

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
                analyze_button.set_enabled(state.fireworks_connected)
                if state.fireworks_connected:
                    progress_label.set_text("Ready to scan the uploaded document.")
                    analysis_status.set_text("Fireworks is connected. Configure and run an AI privacy scan.")
                else:
                    progress_label.set_text("Connect Fireworks to enable AI scanning.")
                    analysis_status.set_text(
                        "Manual redaction is ready. AI scanning remains disabled until Fireworks is connected."
                    )
                invalidate_final_preview()
                ui.notify(
                    "File loaded. Choose a selection mode and mark the content to remove.",
                    type="positive",
                )

            uploader = ui.upload(
                label="Choose PDF or image",
                on_upload=handle_upload,
                auto_upload=True,
                max_file_size=MAX_UPLOAD_BYTES,
            ).props('accept="application/pdf,.pdf,image/jpeg,.jpg,.jpeg,image/png,.png,image/webp,.webp,image/bmp,.bmp"').classes(
                "w-full"
            )
            uploader.enable()
            ui.label(
                "Manual redaction works locally without Fireworks. Connect an API key only to unlock AI suggestions."
            ).classes("muted text-xs")

        editor_controls = ui.row().classes("toolbar w-full items-center gap-2 flex-wrap")
        editor_controls.visible = False

        workspace = ui.element("div").classes("workspace-grid w-full")
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
            undo_button = ui.button("Undo", icon="undo").props("outline")
            redo_button = ui.button("Redo", icon="redo").props("outline")
            clear_button = ui.button("Clear page", icon="delete_outline")
            rectangle_count = ui.label("0 redactions on this page").classes("muted ml-auto")
            ui.label("Ctrl+Z undo · Ctrl+Shift+Z redo").classes("shortcut-chip")

        with workspace:
            with ui.column().classes("editor-column w-full gap-4"):
                with ui.card().classes("glass-card w-full p-5"):
                    ui.label("03 // EDITOR").classes("section-kicker")
                    ui.label("Mark redactions").classes("text-xl font-bold")
                    mode_help = ui.label(
                        "Draw boxes: click and drag over any area you want to remove."
                    ).classes("muted")
                    editor_image = ui.interactive_image(
                        source="",
                        size=(CANVAS_WIDTH, CANVAS_HEIGHT),
                        events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                        cross=True,
                    ).classes("editor-image")

                with ui.card().classes("glass-card w-full p-5"):
                    ui.label("05 // EXPORT").classes("section-kicker")
                    ui.label("Final preview").classes("text-xl font-bold")
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

            with ui.card().classes("ai-sidebar w-full p-4 gap-3"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.column().classes("gap-0"):
                        ui.label("04 // AI REVIEW").classes("section-kicker")
                        ui.label("AI redaction sidebar").classes("text-xl font-bold")
                        ui.label("Connect Fireworks to scan, then click suggestions to select or deselect them.").classes("muted text-sm")
                    ui.icon("shield", size="md").classes("text-teal-300")

                with ui.column().classes("sidebar-section w-full gap-2"):
                    ui.label("SCAN SETTINGS").classes("section-kicker")
                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        analysis_scope = ui.select(
                            options={"all": "All pages", "current": "Current page"},
                            value="all",
                            label="Scan scope",
                        ).props("outlined dense").classes("w-40")
                        run_ocr_checkbox = ui.checkbox("Use OCR", value=True)
                    ui.label(
                        "OCR reads text from scanned PDFs and images so sensitive information can be detected and suggested for redaction."
                    ).classes("muted text-xs -mt-1")

                    custom_instruction = ui.textarea(
                        label="Custom redaction instruction",
                        placeholder="e.g. Redact every link and any picture containing a person.",
                    ).props("outlined autogrow clearable maxlength=700").classes("w-full")
                    ui.label(
                        "Describe extra text or visual targets. Standard privacy checks still run alongside your instruction."
                    ).classes("muted text-xs -mt-1")
                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        all_links_button = ui.button(
                            "All links",
                            on_click=lambda: custom_instruction.set_value("Redact every visible web link or URL."),
                            icon="link",
                        ).props("flat dense")
                        faces_button = ui.button(
                            "Faces / photos",
                            on_click=lambda: custom_instruction.set_value("Redact all faces, portraits and pictures containing people."),
                            icon="face",
                        ).props("flat dense")
                        signatures_button = ui.button(
                            "Signatures",
                            on_click=lambda: custom_instruction.set_value("Redact every handwritten or digital signature."),
                            icon="draw",
                        ).props("flat dense")

                    learn_from_review = ui.checkbox("Use my approvals to calibrate future confidence", value=False)
                    ui.label(
                        "Stores category-level accept/reject metadata only — never document text, images, coordinates or secret values."
                    ).classes("muted text-xs -mt-1")

                    threshold_label = ui.label("Show suggestions at 60% confidence or higher").classes("muted text-sm")
                    confidence_threshold = ui.slider(min=0.5, max=0.99, step=0.01, value=0.60).classes("w-full")
                    analyze_button = ui.button("Run AI privacy scan", icon="auto_awesome").props("color=primary unelevated").classes("w-full")
                    analyze_button.disable()

                    with ui.column().classes("progress-shell w-full gap-2") as scan_progress_shell:
                        with ui.row().classes("w-full items-center justify-between gap-2"):
                            progress_label = ui.label("Connect Fireworks to enable AI scanning.").classes("muted text-xs")
                            progress_percentage = ui.label("0%").classes("text-xs text-cyan-200")
                        scan_progress = ui.linear_progress(value=0).props("rounded color=primary track-color=blue-grey-10").classes("w-full")
                    scan_progress_shell.visible = False
                    analysis_status = ui.label(
                        "Manual redaction is available now. Connect Fireworks to unlock AI scanning."
                    ).classes("scan-status muted text-sm")

                with ui.column().classes("sidebar-section w-full gap-2"):
                    with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                        with ui.column().classes("gap-0"):
                            ui.label("SUGGESTIONS").classes("section-kicker")
                            ui.label("Review and apply").classes("font-semibold")
                        ui.badge("HUMAN APPROVAL REQUIRED").props("outline color=positive")
                    ui.label(
                        "Click anywhere on a suggestion card to select or deselect it. Suggestions remain temporary until applied."
                    ).classes("muted text-xs")
                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        select_all_button = ui.button("Select all", icon="done_all").props("outline dense")
                        clear_selection_button = ui.button("Clear", icon="remove_done").props("outline dense")
                        apply_suggestions_button = ui.button("Apply selected", icon="playlist_add_check").props("color=primary unelevated")
                    ui.separator()
                    suggestions_container = ui.column().classes("suggestion-scroll w-full gap-2")

        def capture_snapshot(label: str = "Edit") -> HistorySnapshot:
            return HistorySnapshot(
                redactions=copy.deepcopy(state.redactions),
                suggestion_states={
                    suggestion.suggestion_id: (suggestion.selected, suggestion.applied)
                    for suggestion in state.ai_suggestions
                },
                current_page=state.current_page,
                label=label,
            )

        def restore_snapshot(snapshot: HistorySnapshot) -> None:
            state.redactions = copy.deepcopy(snapshot.redactions)
            for suggestion in state.ai_suggestions:
                selected, applied = snapshot.suggestion_states.get(
                    suggestion.suggestion_id, (suggestion.selected, suggestion.applied)
                )
                suggestion.selected = selected
                suggestion.applied = applied
            state.current_page = min(max(snapshot.current_page, 0), max(state.page_count - 1, 0))
            invalidate_final_preview()

        def remember_change(before: HistorySnapshot) -> None:
            state.undo_stack.append(before)
            if len(state.undo_stack) > 100:
                state.undo_stack.pop(0)
            state.redo_stack.clear()
            update_history_buttons()

        def update_history_buttons() -> None:
            undo_button.set_enabled(bool(state.undo_stack))
            redo_button.set_enabled(bool(state.redo_stack))

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
                    'fill="#02080d" fill-opacity="0.82" stroke="#5eead4" stroke-width="2" />'
                )

            threshold = float(confidence_threshold.value or 0.60)
            for suggestion in state.ai_suggestions:
                if suggestion.page_index != state.current_page or suggestion.applied or suggestion.confidence < threshold:
                    continue
                selected = suggestion.selected
                stroke = "#5eead4" if selected else "#6b8792"
                opacity = "0.20" if selected else "0.06"
                for x0, y0, x1, y1 in suggestion.rects:
                    left, top = state.page_view.page_to_canvas(min(x0, x1), min(y0, y1))
                    right, bottom = state.page_view.page_to_canvas(max(x0, x1), max(y0, y1))
                    shapes.append(
                        f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" '
                        f'fill="#5eead4" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="2" stroke-dasharray="6 4" />'
                    )

            if state.draft_rect is not None:
                x0, y0, x1, y1 = state.draft_rect
                left, top = state.page_view.page_to_canvas(min(x0, x1), min(y0, y1))
                right, bottom = state.page_view.page_to_canvas(max(x0, x1), max(y0, y1))
                if selection_mode.value == "text":
                    fill = "#7dd3fc"
                    opacity = "0.18"
                else:
                    fill = "black"
                    opacity = "0.38"
                shapes.append(
                    f'<rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" '
                    f'fill="{fill}" fill-opacity="{opacity}" stroke="#a78bfa" stroke-width="2" stroke-dasharray="8 5" />'
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
                    card_classes = "suggestion-card w-full p-3"
                    if suggestion.selected and not suggestion.applied:
                        card_classes += " suggestion-card-selected"
                    if suggestion.applied:
                        card_classes += " suggestion-card-applied"

                    def toggle_suggestion(item=suggestion) -> None:
                        if item.applied or state.scan_in_progress:
                            return
                        item.selected = not item.selected
                        render_suggestions()
                        refresh_overlay()

                    with ui.card().classes(card_classes).on("click", toggle_suggestion):
                        with ui.row().classes("w-full items-start gap-2"):
                            checkbox = ui.checkbox(value=suggestion.selected and not suggestion.applied)
                            checkbox.set_enabled(not suggestion.applied and not state.scan_in_progress)
                            checkbox.on("click", js_handler="event.stopPropagation()")

                            def update_selection(event: events.ValueChangeEventArguments, item=suggestion) -> None:
                                if item.applied or state.scan_in_progress:
                                    return
                                item.selected = bool(event.value)
                                render_suggestions()
                                refresh_overlay()

                            checkbox.on_value_change(update_selection)
                            with ui.column().classes("grow gap-0"):
                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                    ui.badge(suggestion.category_label)
                                    ui.label(f"{suggestion.confidence * 100:.0f}% • Page {suggestion.page_index + 1}").classes(
                                        "font-semibold confidence-glow"
                                    )
                                    if suggestion.applied:
                                        ui.badge("Applied").props("color=positive")
                                ui.label(suggestion.preview).classes("text-sm break-all")
                                ui.label(f"{suggestion.reason} Source: {suggestion.source}.").classes(
                                    "muted text-xs"
                                )
                                if suggestion.page_index != state.current_page:
                                    show_page_button = ui.button(
                                        "Show page",
                                        icon="find_in_page",
                                        on_click=lambda page=suggestion.page_index: show_page(page),
                                    ).props("flat dense").classes("self-start")
                                    show_page_button.on("click", js_handler="event.stopPropagation()")

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
                before = capture_snapshot("Manual redaction")
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
                    remember_change(before)
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

        async def undo_redaction() -> None:
            if not state.undo_stack:
                ui.notify("Nothing to undo.", type="info")
                return
            current = capture_snapshot("Redo")
            snapshot = state.undo_stack.pop()
            state.redo_stack.append(current)
            restore_snapshot(snapshot)
            await show_page(snapshot.current_page)
            render_suggestions()
            refresh_overlay()
            update_history_buttons()
            ui.notify(f"Undid: {snapshot.label}", type="info")

        async def redo_redaction() -> None:
            if not state.redo_stack:
                ui.notify("Nothing to redo.", type="info")
                return
            current = capture_snapshot("Undo")
            snapshot = state.redo_stack.pop()
            state.undo_stack.append(current)
            restore_snapshot(snapshot)
            await show_page(snapshot.current_page)
            render_suggestions()
            refresh_overlay()
            update_history_buttons()
            ui.notify("Redid the last edit.", type="info")

        def clear_page_redactions() -> None:
            if current_rectangles():
                before = capture_snapshot("Clear page")
                current_rectangles().clear()
                remember_change(before)
                invalidate_final_preview()
                refresh_overlay()

        async def keyboard_shortcuts(event: events.KeyEventArguments) -> None:
            if not event.action.keydown or event.action.repeat:
                return
            command = event.modifiers.ctrl or event.modifiers.meta
            key_name = event.key.name.lower()
            if command and key_name == "z":
                if event.modifiers.shift:
                    await redo_redaction()
                else:
                    await undo_redaction()
            elif command and key_name == "y":
                await redo_redaction()

        ui.keyboard(on_key=keyboard_shortcuts, repeating=False)
        undo_button.on_click(undo_redaction)
        redo_button.on_click(redo_redaction)
        clear_button.on_click(clear_page_redactions)
        update_history_buttons()

        def set_scan_progress(value: float, text: str) -> None:
            value = max(0.0, min(float(value), 1.0))
            scan_progress.set_value(value)
            progress_percentage.set_text(f"{round(value * 100):d}%")
            progress_label.set_text(text)

        def set_scan_controls_locked(locked: bool) -> None:
            state.scan_in_progress = locked
            controls = [
                fireworks_api_key,
                vision_model_select,
                connect_button,
                analysis_scope,
                run_ocr_checkbox,
                custom_instruction,
                all_links_button,
                faces_button,
                signatures_button,
                learn_from_review,
                confidence_threshold,
            ]
            for control in controls:
                control.disable() if locked else control.enable()

            if locked:
                uploader.disable()
                analyze_button.disable()
                apply_suggestions_button.disable()
                select_all_button.disable()
                clear_selection_button.disable()
            else:
                uploader.enable()
                analyze_button.set_enabled(
                    state.fireworks_connected and state.file_bytes is not None
                )
                apply_suggestions_button.enable()
                select_all_button.enable()
                clear_selection_button.enable()

        async def scan_for_sensitive_information() -> None:
            if state.scan_in_progress:
                return
            if not state.fireworks_connected or not state.connected_api_key or not state.connected_model:
                ui.notify("Connect and validate a Fireworks API key before running a scan.", type="warning")
                return
            if state.file_bytes is None or state.kind is None:
                ui.notify("Upload a PDF or image first.", type="warning")
                return

            entered_api_key = str(fireworks_api_key.value or "").strip()
            selected_model = str(vision_model_select.value or DEFAULT_UI_FIREWORKS_MODEL).strip()
            if (
                entered_api_key != state.connected_api_key
                or selected_model != state.connected_model
            ):
                invalidate_fireworks_connection()
                ui.notify("The API key or model changed. Press Connect again before scanning.", type="warning")
                return

            if analysis_scope.value == "current":
                page_indexes = [state.current_page]
                retained_suggestions = [
                    suggestion
                    for suggestion in state.ai_suggestions
                    if suggestion.page_index != state.current_page
                ]
            else:
                page_indexes = list(range(state.page_count))
                retained_suggestions = []

            previous_suggestions = state.ai_suggestions
            scanned_suggestions: list[SensitiveSuggestion] = []
            warnings: list[str] = []
            total_tokens = 0
            total_pages = max(len(page_indexes), 1)

            set_scan_controls_locked(True)
            scan_progress_shell.visible = True
            set_scan_progress(0.0, "Preparing the scan…")
            analysis_status.set_text("AI scan in progress. Connection and model settings are locked.")

            try:
                for position, page_index in enumerate(page_indexes, start=1):
                    page_start = (position - 1) / total_pages
                    page_span = 1 / total_pages
                    set_scan_progress(
                        page_start + page_span * 0.08,
                        f"Rendering page {page_index + 1} ({position}/{total_pages})…",
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

                    set_scan_progress(
                        page_start + page_span * 0.25,
                        f"Running OCR, local detectors and Fireworks vision on page {page_index + 1}…",
                    )
                    result = await run.io_bound(
                        analyze_page,
                        page_index=page_index,
                        image=image,
                        view=view,
                        embedded_words=words,
                        use_ai=True,
                        run_ocr=bool(run_ocr_checkbox.value),
                        custom_instruction=str(custom_instruction.value or "").strip(),
                        fireworks_api_key=state.connected_api_key,
                        fireworks_model=state.connected_model,
                    )
                    if not result.ai_used:
                        raise RuntimeError(
                            "Fireworks vision did not complete, so no local-only scan was accepted."
                        )
                    scanned_suggestions.extend(result.suggestions)
                    warnings.extend(result.warnings)
                    total_tokens += result.token_count
                    set_scan_progress(
                        position / total_pages,
                        f"Completed page {page_index + 1} ({position}/{total_pages}).",
                    )

                state.ai_suggestions = retained_suggestions + scanned_suggestions
                render_suggestions()
                refresh_overlay()
                visible_count = len(visible_suggestions())
                model_label = FIREWORKS_MODEL_CATALOG[state.connected_model]["label"].split(" — ", 1)[0]
                status = (
                    f"Found {len(scanned_suggestions)} new suggestion(s); {visible_count} meet the current threshold. "
                    f"Analysed {total_tokens} text token(s) using {model_label}."
                )
                analysis_status.set_text(status)
                set_scan_progress(1.0, "Scan complete — review the suggestions below.")
                if warnings:
                    ui.notify(warnings[0], type="warning", timeout=10000)
                else:
                    ui.notify("Sensitive-information scan complete.", type="positive")
            except Exception as exc:
                state.ai_suggestions = previous_suggestions
                render_suggestions()
                refresh_overlay()
                analysis_status.set_text("The scan failed. No partial scan results were applied.")
                set_scan_progress(0.0, "Scan failed — check the connection message and try again.")
                ui.notify(f"AI scan failed: {exc}", type="negative", timeout=12000)
            finally:
                set_scan_controls_locked(False)

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

            before = capture_snapshot("Apply AI suggestions")
            reviewed = [suggestion for suggestion in visible_suggestions() if not suggestion.applied]
            added_rectangles = 0
            for suggestion in chosen:
                page_rectangles = state.redactions.setdefault(suggestion.page_index, [])
                for rect in suggestion.rects:
                    if not any(rectangles_match(rect, existing) for existing in page_rectangles):
                        page_rectangles.append(rect)
                        added_rectangles += 1
                suggestion.applied = True
                suggestion.selected = False

            remember_change(before)
            if bool(learn_from_review.value):
                try:
                    record_review_metadata(
                        filename=state.filename,
                        custom_prompt_used=bool((custom_instruction.value or "").strip()),
                        rows=[
                            {
                                "category": suggestion.category,
                                "accepted": suggestion in chosen,
                                "confidence": suggestion.confidence,
                                "source": suggestion.source,
                            }
                            for suggestion in reviewed
                        ],
                    )
                except Exception as exc:
                    ui.notify(f"Could not save local review calibration: {exc}", type="warning")
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

        with ui.card().classes("glass-card w-full p-5"):
            ui.label("SECURITY NOTES").classes("section-kicker")
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
