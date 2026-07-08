# Future Vision Fine-Tuning Plan

The application currently improves detection with deterministic rules, OCR, a detailed category playbook, synthetic few-shot examples and optional category-level confidence calibration. This is safer and immediately usable, but it is not weight training.

## Recommended dataset

Build a consented, synthetic or properly licensed dataset of document/page images. Each example should contain:

- the source image or rendered PDF page;
- the detection instruction;
- category labels;
- exact token IDs where text extraction exists;
- normalized bounding boxes for visual/scanned regions;
- difficult negative examples;
- accepted and rejected findings.

Do not use real private documents without explicit authorisation. Replace secrets with synthetic values.

## Coverage

Balance examples across emails, phones, addresses, names, usernames, account/card numbers, credentials, connection strings, private keys, IPs, paths, sensitive URLs, student/employee IDs, dates of birth, chats, authentication codes and QR codes. Include scans, screenshots, CVs, forms, letters, source-code screenshots and rotated/low-resolution pages.

## Fireworks path

Fireworks provides supervised fine-tuning for vision-language models using image and text data. At the time this project was updated, Fireworks documentation listed the Qwen 2.5 VL family for VLM fine-tuning. A fine-tuned LoRA is deployed through an on-demand deployment rather than the shared serverless endpoint.

The starter JSONL file in `training_data/` demonstrates text-only message structure. A production VLM dataset must additionally reference images and include accurate bounding-box outputs in the format required by the chosen Fireworks training workflow.
