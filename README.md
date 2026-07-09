# Aurora Document & Image Redactor

Aurora is a Python/NiceGUI application for permanent manual and AI-assisted redaction of PDFs and image files. It uses a dark pearl-blue/green cybersecurity interface and keeps every AI finding reviewable before it becomes a redaction.

## Current capabilities

- Upload PDFs, PNG, JPEG, WebP and BMP files.
- Draw permanent redaction boxes manually without an API key.
- Select real embedded PDF text and redact exact word regions.
- Undo with `Ctrl+Z`; redo with `Ctrl+Shift+Z` or `Ctrl+Y`.
- Preview the rewritten output before downloading it.
- Enter and validate a Fireworks API key in the UI.
- Choose MiniMax M3, Qwen 3.7 Plus or Kimi K2.6.
- Run OCR on scanned pages and images.
- Review confidence-scored AI suggestions in a scrollable sidebar.
- Click suggestion cards or checkboxes to select and deselect them.
- Apply only the selected suggestions.
- Add a custom instruction, including strict instructions such as `Only redact phone numbers`.

## Important detection behaviour

The Fireworks vision model is the sole classifier for AI PII suggestions. OCR and embedded PDF extraction supply text and exact coordinates, but regex-based PII findings are not merged into AI scan results.

Local QR-code localisation may still propose precise QR regions because it is a computer-vision detector rather than a text-pattern detector.

### Blank custom instruction

Leaving the custom instruction blank runs the complete privacy scan.

### Additive custom instruction

An ordinary instruction adds a custom target while retaining the normal privacy scan:

```text
Redact all faces and vehicle registration plates.
```

### Exclusive custom instruction

Words such as `only`, `just`, `exclusively` or `solely` restrict the result categories:

```text
Only redact phone numbers.
```

Aurora reinforces this rule in the model prompt and filters the returned results again before showing them, preventing unrelated categories from appearing.

## Supported privacy categories

- Email addresses
- Phone numbers
- Home and postal addresses
- Usernames
- Full personal names
- Account numbers
- Bank-card-like numbers
- API keys
- Access tokens
- Passwords and password-like values
- Database connection strings
- Private keys
- IP addresses
- File paths
- URLs containing sensitive query parameters
- Student IDs
- Employee IDs
- Dates of birth
- Private chat messages and message panels
- Authentication, OTP and MFA codes
- QR codes
- Ordinary links when requested
- Faces and photographs when requested
- Arbitrary custom text or visual regions

## Fireworks models

| Model | Relative strength | Indicative serverless billing per 1M tokens | Notes |
|---|---:|---:|---|
| MiniMax M3 | Medium | $0.30 input / $0.06 cached / $1.20 output | Lowest-cost default for routine forms and screenshots |
| Qwen 3.7 Plus | Strong | $0.40 input / $0.08 cached / $1.60 output | Recommended quality-to-cost balance |
| Kimi K2.6 | Strong | $0.95 input / $0.16 cached / $4.00 output | Premium model for difficult pages; usually slower |

Prices are indicative and can change. Check the Fireworks model library before production use.

Aurora uses model-specific scan settings. Kimi K2.6 runs with low reasoning effort, a smaller response budget and a longer timeout to reduce unnecessary delay.

## Progress behaviour

Fireworks does not expose an exact page-analysis percentage. During a model request, Aurora therefore shows an elapsed-time heartbeat and advances conservatively within the current page stage. This confirms the request is still active without falsely claiming that the model is nearly finished.

The scan now uses one JSON-mode Fireworks request per page instead of repeatedly trying several long response formats. If the request fails, the old suggestion set is restored and partial new results are discarded.

## Accuracy improvements

- The page image is sent at up to 2048 pixels on its longest side by default.
- OCR and embedded-text tokens include normalised coordinates as well as token IDs.
- The model can return token IDs for exact text boxes or visual bounding boxes for scanned and graphical regions.
- Names and addresses receive explicit contextual instructions and synthetic examples.
- Suggestion counts show how many results are visible and how many are hidden by the confidence threshold.
- Local approval calibration is only applied when the user enables it.

## Install and run on Windows

Open the project folder in VS Code and run:

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

Manual redaction works immediately. AI scanning stays disabled until a Fireworks API key and model are validated through the **Connect API key** button.

## Optional runtime settings

The API key is entered in the UI and is not read from `.env`. The `.env` file can contain local runtime settings:

```env
FIREWORKS_IMAGE_MAX_SIDE=2048
HOST=0.0.0.0
PORT=8081
```

The entered API key is held only in the current page session and is not written to disk by Aurora.

## Basic workflow

1. Upload a PDF or image.
2. Add any manual boxes or embedded-text selections.
3. Optionally connect Fireworks and choose a model.
4. Enable OCR for scans, screenshots or image files.
5. Leave the instruction blank for a full scan, or type a custom scope.
6. Run the AI scan.
7. Review the suggestion cards.
8. Select only the suggestions you want.
9. Press **Apply selected**.
10. Generate the final preview.
11. Inspect and download the permanently rewritten output.

## Project structure

```text
.
├── app.py
├── ai_service.py
├── pdf_service.py
├── redaction_knowledge.py
├── feedback_store.py
├── requirements.txt
├── README.md
├── TRAINING.md
├── Dockerfile
├── compose.yaml
├── .env.example
└── training_data/
    └── redaction_examples.jsonl
```

### Main files

- `app.py`: NiceGUI interface, keyboard history, scan progress and suggestion review.
- `ai_service.py`: OCR, token mapping, QR localisation, Fireworks request and result parsing.
- `pdf_service.py`: rendering, coordinates and permanent PDF/image redaction.
- `redaction_knowledge.py`: category definitions and synthetic in-context examples.
- `feedback_store.py`: optional category-level review calibration without storing document contents.
- `TRAINING.md`: future vision-model fine-tuning guidance.

## Privacy and security

Running an AI scan sends the rendered page and extracted text tokens to Fireworks using the key entered by the user. Manual redaction, preview and export can be used without sending the document to Fireworks.

Do not hard-code or commit API keys. Use HTTPS before deploying Aurora remotely. Only process documents that you are authorised to send to an external AI provider.

AI, OCR and visual bounding boxes can produce false positives or false negatives. Always inspect every page of the final exported file before sharing it.

## Current limitations

- Password-protected PDFs are not supported.
- Handwriting and very small text can be missed.
- Serverless models can experience latency or capacity variation.
- Kimi K2.6 can remain slower than the smaller options even with low reasoning effort.
- Model bounding boxes may be approximate.
- Custom requests depend on the selected model understanding the requested target.
- This prototype is not a certified legal, compliance or classified-document redaction system.
