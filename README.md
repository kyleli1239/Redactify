# Aurora Document & Image Redactor

A Python/NiceGUI application for manual and AI-assisted redaction of PDFs and ordinary image files. The interface uses a dark pearl-blue/green aurora theme, while every AI finding stays reviewable before it becomes a permanent redaction.

## What is new

- Dark glass-style cybersecurity interface with rounded panels and an aurora blue/green palette.
- **Ctrl+Z** undo for manual redactions, text selections, clearing a page and applying AI suggestions.
- **Ctrl+Shift+Z** or **Ctrl+Y** redo.
- A mandatory session-only Fireworks API-key connection step with explicit validation before the application unlocks.
- A Fireworks vision-model dropdown showing task strength, usage style and indicative serverless pricing. Changing the key or model invalidates the connection until it is checked again.
- A live page-by-page scan progress bar. API credentials and model selection are locked while a scan is running.
- A fully scrollable review sidebar split into separate scan-settings and suggestion sections.
- A custom AI instruction box, for example:
  - `Redact every visible web link.`
  - `Redact all faces and photographs containing people.`
  - `Redact every signature and vehicle registration plate.`
- Quick prompt presets for links, faces/photos and signatures.
- Local URL matching when the user requests all links.
- Local OpenCV face proposals when the user requests faces or pictures of people; the Fireworks vision model also checks the full page.
- A reusable privacy knowledge pack covering all supported categories.
- Synthetic few-shot examples supplied to the model on each scan.
- Optional privacy-preserving local confidence calibration from accepted/rejected suggestions.
- A starter JSONL dataset under `training_data/` for a future supervised fine-tuning workflow.

## Detection pipeline

1. PyMuPDF extracts embedded PDF text and exact word coordinates.
2. RapidOCR reads scanned PDFs and images when **Use OCR** is enabled.
3. Deterministic detectors catch rigid formats such as emails, tokens, IPs, IDs, cards, paths and authentication codes.
4. Context detectors propose names and multi-line postal addresses.
5. OpenCV detects QR codes and, when requested, face-like regions.
6. The rendered page, token IDs, custom instruction, category playbook and compact examples are sent to the configured Fireworks vision model.
7. The scan reports live page-by-page progress while the API key and model controls remain locked.
8. Findings appear as confidence-scored suggestions in the independently scrollable sidebar.
9. Only selected suggestions become redaction regions.
10. The final PDF or image is permanently rewritten before download.

## Supported privacy categories

- email addresses
- phone numbers
- home addresses where reasonably detectable
- usernames
- full names where confidently detected
- account numbers
- bank card-like numbers
- API keys
- access tokens
- passwords or password-like fields
- database connection strings
- private keys
- IP addresses
- file paths
- URLs containing sensitive query parameters
- student IDs
- employee IDs
- dates of birth
- private chat messages or message panels
- authentication codes
- QR codes

The custom instruction can additionally request ordinary links, faces/photos, signatures, logos, number plates or other visible/textual targets.

## Run on Windows in VS Code

From the folder containing `app.py`:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:8081
```

Enter a Fireworks API key in the **Fireworks Access** section, choose a model and press **Connect API key**. Aurora sends a deliberately tiny validation request to confirm both the credentials and access to the selected model. The check may incur negligible token usage.

The key is kept only in the current page session and is not written to disk by Aurora. Uploading and AI scanning remain locked until validation succeeds. There is no `.env` fallback and no local-only scan mode in this build. Editing either the key or selected model immediately invalidates the connection and requires another connection check.

For a remotely hosted deployment, use HTTPS before allowing users to enter API keys. Never place real API keys in source code, README files, screenshots or Git commits.


## Fireworks model selector

The Fireworks Access section includes a curated list of serverless models that accept image input. The selected model is validated together with the key and is then used for every scan until the connection settings change.

| Model | Task strength | Indicative serverless billing per 1M tokens | Best use |
|---|---|---|---|
| MiniMax M3 | Medium | $0.30 input / $0.06 cached / $1.20 output | Lowest-cost default for routine forms, screenshots and documents |
| Qwen 3.7 Plus | Strong | $0.40 input / $0.08 cached / $1.60 output | Recommended quality-to-cost balance for complex layouts and contextual PII |
| Kimi K2.6 | Strong | $0.95 input / $0.16 cached / $4.00 output | Premium multimodal reasoning for difficult or ambiguous pages |

Image content is billed as input tokens. Prices are informational values checked in July 2026 and may change; confirm current rates in the Fireworks model library before production use.

The strength rating is relative to this redaction-review task, not a general benchmark score. Local OCR, regular-expression detectors and QR detection run alongside the selected model, but their results are not accepted as a standalone scan when Fireworks is unavailable.

## Connection and scan behaviour

- The upload control is disabled until the Fireworks connection succeeds.
- The **Run AI privacy scan** button is disabled until the connection succeeds.
- Invalid or missing keys cannot fall back to local-only detection.
- The API-key field, model selector and scan settings are disabled while a scan is active.
- A live progress bar reports page rendering, OCR/detection and completion.
- Scan results are staged and only replace the previous suggestion set after the full scan succeeds.
- If a scan fails, partial results are discarded.
- The AI review sidebar has its own vertical scrollbar, so long status messages and suggestion lists remain accessible.

## Custom AI instructions

The instruction is treated as an additional target; it does not replace the standard privacy scan. Text findings use exact OCR/PDF token boxes when possible. Visual findings use bounding boxes.

Examples:

```text
Redact all visible links, including ordinary public URLs.
Redact all profile pictures, faces and photographs containing people.
Redact signatures, handwritten initials and passport photographs.
Redact every mention of Project Aurora and its logo.
```

## About the included “training” data

`redaction_knowledge.py` and `training_data/redaction_examples.jsonl` improve inference through stronger instructions and few-shot examples. They **do not change the model’s weights**.

The optional local review setting stores only category, confidence, source and accept/reject metadata in `data/redaction_feedback.jsonl`. It never stores document text, images, coordinates or secret previews. Once enough reviews exist, the app applies a very small category-level confidence calibration.

Actual vision-model fine-tuning must be run as a separate Fireworks training job using an image-and-text labelled dataset and a model supported for VLM fine-tuning. Fine-tuned LoRA models require an on-demand deployment rather than Fireworks serverless inference. See `TRAINING.md` for the recommended future path.

## Main files

- `app.py` — themed NiceGUI interface, keyboard history, sidebar and custom prompt
- `ai_service.py` — OCR, local detectors, custom prompt handling and Fireworks vision integration
- `redaction_knowledge.py` — category playbook and synthetic in-context examples
- `feedback_store.py` — privacy-preserving local confidence calibration
- `training_data/redaction_examples.jsonl` — starter synthetic dataset
- `pdf_service.py` — rendering, coordinates and permanent PDF/image redaction
- `.env.example` — optional host, port and image-size runtime settings; API keys are entered only in the UI

## Privacy

Connecting sends a tiny test completion to Fireworks. Running the Fireworks scan sends the rendered page, extracted text tokens and the selected model ID using the validated key. Aurora does not intentionally log or save the entered key. Only process documents you are authorised to send to a third-party API. Always review suggestions and the final preview: OCR, heuristics and vision models can miss content or produce false positives.
