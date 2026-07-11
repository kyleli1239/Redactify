# Redactify

Redactify is an AI-assisted document and image redaction tool designed to help users permanently remove sensitive content and personally identifiable information (PII) from PDFs and images. The application combines local document processing with Fireworks AI vision analysis to detect likely private information, then lets users review and manually apply redactions with full control. 

It is built on the Fireworks AI API, leveraging open-source large language models accelerated by AMD GPU compute resources. Every document analysis request is processed through Fireworks' AMD-powered inference infrastructure, enabling fast, scalable and reliable extraction of sensitive data without requiring users to manage their own AI hardware

---

## Hackathon context

Redactify was built for the AMD Developer Hackathon: ACT II — Track 3 Unicorn, July 6–11, 2026.

The hackathon rules require projects to demonstrate AMD resource usage, and Redactify is explicitly designed around that requirement. The app showcases compute-heavy document workflows including PDF rendering, OCR, image analysis, and AI-powered vision inference in a containerized architecture that is ready to run on AMD-powered infrastructure. This makes the project more than a simple front-end demo: it demonstrates meaningful compute utilization through real document-processing workloads.

---

## Why This Exists

Manually redacting PDFs is tedious, error-prone, and doesn't scale. Lawyers, HR teams, healthcare administrators, and researchers spend hours hunting through documents to black out sensitive data before sharing them. **Redactify** automates this in one click, combining the speed of AMD-accelerated inference with the precision of a large language model.

---

## What Redactify does

* Upload a PDF or image file
* Draw boxes or select embedded PDF text to redact sensitive regions
* Run AI-assisted scanning for emails, phone numbers, addresses, names, credentials, QR codes, and other sensitive patterns
* Preview the final redacted output before downloading it
* Remove metadata, hidden text, attachments, and JavaScript from PDFs where possible

Redactify supports:

* PDF documents
* PNG images
* JPEG images
* WebP images
* BMP images
* Multi-page PDFs
* Rotated PDF pages
* Scanned documents
* Screenshots and photographs of documents

The default maximum upload size is **50 MB**.

The AI never immediately modifies the uploaded file. Findings first appear in the review sidebar with:

* Category
* Confidence percentage
* Masked preview
* Detection source
* Reason for the suggestion
* Selection checkbox

Users can select individual suggestions, select all visible suggestions or clear the current selection before applying anything.

### Manual Redaction

AI-assisted redaction speeds up identifying sensitive information, but it isn’t flawless. Fireworks AI may occasionally miss details or flag content incorrectly, so every detection includes a confidence score that users can review and adjust. A full manual redaction mode is also provided, ensuring users always have precise control over what is removed.

* Click-and-drag redaction boxes
* Embedded PDF text selection
* Page-by-page navigation
* Clear-page controls
* Final output preview

Embedded text selection snaps to the real word positions stored inside a PDF. Box drawing works with PDFs, scans, graphics and ordinary image files.

### Undo and redo

Redaction edits can be reversed using the interface buttons or keyboard shortcuts:

```text
Ctrl + Z          Undo
Ctrl + Shift + Z  Redo
Ctrl + Y          Redo
```

Manual redaction is also useful if a Fireworks API key is missing allowing you to still redact PPI without using the AI assistance tool

---

## AI processing workflow

```text
Upload PDF or image
        ↓
Extract embedded PDF text
        ↓
Optionally run local OCR
        ↓
Send the rendered page and extracted tokens to Fireworks
        ↓
Display confidence-scored suggestions
        ↓
User approves or rejects suggestions
        ↓
Apply approved redaction regions
        ↓
Generate final preview
        ↓
Download permanently redacted output
```

## Technology stack

* Python
* NiceGUI
* PyMuPDF
* Pillow
* OpenAI
* OpenCV
* RapidOCR
* NumPy
* Pydantic
* OpenAI-compatible Fireworks client
* Docker

---

## Local setup

Requirements:

- Python 3.10+
- pip

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
python app.py
```

Then open http://localhost:8081 in your browser.

## Run with Docker

```bash
docker compose up --build
```

The app will be available at http://localhost:8081.

---

## Using Fireworks AI

To use Fireworks AI, an API key is needed https://fireworks.ai/

1. Run the application through ```python app.py```
2. Open http://localhost:8081 in your browser (this should happen automatically)
3. Enter your Fireworks API key into the API credential textbox
4. Select a model (MiniMax M3 is recommended if API credits are limited)
5. Click the "Connect API Key" button
6. If the key is valid, it will say "API key connected successfully — minimax-m3 is ready."
7. Upload a document
8. An editor box should appear
9. On the AI redaction sidebar, click on "Run AI privacy scan"
10. The AI should return suggested redactions. Manually review and approve these suggestions
11. Export and download redacted file

---

## Confidence threshold

The confidence slider controls which AI suggestions are shown.

A lower value increases recall but may show more false positives. A higher value reduces noise but may hide uncertain sensitive information.

Suggested starting points:

```text
0.55–0.70  Higher recall and more manual review
0.70–0.85  Balanced review
0.85+  Only high-confidence suggestions
```

The confidence score is a review aid, not a guarantee that a finding is correct.

---

## Detection knowledge and examples

`redaction_knowledge.py` contains:

* Category definitions
* Positive examples
* False-positive guidance
* Visual detection guidance
* Synthetic few-shot examples

These examples are included in the model instruction to make results more consistent.

The file:

```text
training_data/redaction_examples.jsonl
```

contains starter synthetic examples for future experimentation.

These files do not train the model automatically. They provide rules and in-context examples during inference.

See `TRAINING.md` for a longer-term vision-model fine-tuning plan.

---

## Current limitations

This is just a working prototype. Some features have not been fully developed.

* Password-protected PDFs are not supported.
* OCR accuracy depends on image quality.
* Handwriting detection is limited.
* Face detection has not been fully developed
* Vision-model bounding boxes may be approximate.
* AI confidence is not a security guarantee.
* Malformed PDFs may prevent optional deep sanitisation.
* The model can miss context-dependent information.
