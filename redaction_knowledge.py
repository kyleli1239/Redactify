"""Reusable privacy-detection guidance and synthetic examples.

This file does not modify model weights. It supplies a consistent detection playbook
and compact few-shot examples to the vision model on every scan.
"""

from __future__ import annotations

import json

CATEGORY_PLAYBOOK: dict[str, dict[str, object]] = {
    "email_address": {
        "description": "Personal or work email addresses.",
        "examples": ["alex.taylor@example.com", "support+case@company.co.uk"],
        "avoid": ["generic text saying 'email address' without a value"],
    },
    "phone_number": {
        "description": "Mobile, landline, international and extension-bearing phone numbers.",
        "examples": ["+44 7700 900123", "020 7946 0123", "+1 (415) 555-0123 ext 9"],
        "avoid": ["short unlabeled numbers that are clearly prices or page numbers"],
    },
    "home_address": {
        "description": "Residential or postal addresses, including multi-line addresses.",
        "examples": ["Flat 4, 18 King Street, Coventry, CV1 2AB", "221B Baker Street, London NW1 6XE"],
        "avoid": ["company headquarters unless the user explicitly asks for business addresses"],
    },
    "username": {
        "description": "Account handles, login names, social media usernames and gamer tags.",
        "examples": ["@alex_dev", "username: rkyawzay", "discord: bluefox_17"],
        "avoid": ["ordinary words beginning with @ in prose"],
    },
    "full_name": {
        "description": "Complete personal names in forms, CVs, letters, profiles, signatures or messages.",
        "examples": ["Richmond Kyawzay", "Dr Amelia Hart", "Account holder: Jordan Lee"],
        "avoid": ["company, product, team and organisation names"],
    },
    "account_number": {
        "description": "Bank, utility, customer, membership or service account identifiers.",
        "examples": ["Account No: 12345678", "Customer account 009184726"],
        "avoid": ["invoice totals and ordinary quantities"],
    },
    "bank_card_number": {
        "description": "Credit/debit card-like primary account numbers, usually 13–19 digits.",
        "examples": ["4111 1111 1111 1111", "5500-0000-0000-0004"],
        "avoid": ["long IDs that fail contextual and checksum checks"],
    },
    "api_key": {
        "description": "Provider API keys and secret key values.",
        "examples": ["sk_live_...", "AIza...", "fw_...", "xoxb-..."],
        "avoid": ["the literal words API key with no value"],
    },
    "access_token": {
        "description": "Bearer tokens, JWTs, refresh tokens, session tokens and OAuth credentials.",
        "examples": ["Authorization: Bearer eyJ...", "refresh_token=..."],
        "avoid": ["public non-secret identifiers"],
    },
    "password": {
        "description": "Passwords, PINs, passphrases and password-like values next to credential labels.",
        "examples": ["Password: Winter!2026", "PIN 4829", "passphrase=correct horse..."],
        "avoid": ["masked placeholders such as •••••••• unless their location itself should be hidden"],
    },
    "database_connection_string": {
        "description": "Database DSNs and connection strings containing hosts, users or credentials.",
        "examples": ["postgresql://user:pass@host/db", "mongodb+srv://user:pass@cluster/db"],
        "avoid": ["a database product name without connection details"],
    },
    "private_key": {
        "description": "PEM/OpenSSH private keys and complete private-key blocks.",
        "examples": ["-----BEGIN PRIVATE KEY-----", "-----BEGIN OPENSSH PRIVATE KEY-----"],
        "avoid": ["public keys unless explicitly requested"],
    },
    "ip_address": {
        "description": "IPv4 and IPv6 addresses, especially internal or administrative endpoints.",
        "examples": ["192.168.1.20", "2001:db8::8a2e:370:7334"],
        "avoid": ["version numbers that merely resemble an IP"],
    },
    "file_path": {
        "description": "Windows, Unix, macOS, network-share and cloud-storage paths.",
        "examples": [r"C:\\Users\\alex\\secrets.txt", "/home/alex/.ssh/id_rsa", r"\\server\\share\\private"],
        "avoid": ["short slash-separated prose that is not a path"],
    },
    "sensitive_url": {
        "description": "URLs carrying secrets, sessions, signatures, tokens, reset links or one-time codes.",
        "examples": ["https://site/reset?token=...", "https://storage/file?X-Amz-Signature=..."],
        "avoid": ["ordinary public links unless the custom instruction asks to redact all links"],
    },
    "student_id": {
        "description": "Student, candidate, enrolment and university identifiers.",
        "examples": ["Student ID: 2310456", "Candidate number 00821"],
        "avoid": ["course codes and module numbers"],
    },
    "employee_id": {
        "description": "Employee, payroll, staff and personnel identifiers.",
        "examples": ["Employee ID: E-10482", "Payroll No 716293"],
        "avoid": ["department codes without a person association"],
    },
    "date_of_birth": {
        "description": "Dates explicitly or strongly contextually identified as a person's birth date.",
        "examples": ["DOB: 29/06/2007", "Date of birth 6 June 1998"],
        "avoid": ["invoice dates, issue dates and ordinary historical dates"],
    },
    "private_chat": {
        "description": "Private DMs, SMS messages, messaging bubbles, chat panels and conversation excerpts.",
        "examples": ["WhatsApp message bubbles", "Discord direct-message panel", "SMS conversation screenshot"],
        "avoid": ["public article comments unless the custom instruction requests them"],
    },
    "authentication_code": {
        "description": "OTP, MFA, verification, recovery and one-time authentication codes.",
        "examples": ["Your code is 381204", "MFA: 842 119", "backup code A7K2-P9Q4"],
        "avoid": ["ordinary unlabeled short numbers"],
    },
    "qr_code": {
        "description": "Every visible QR code, whether or not its payload is readable.",
        "examples": ["login QR", "payment QR", "ticket QR"],
        "avoid": ["ordinary square logos"],
    },
    "general_url": {
        "description": "Any visible web link when the custom instruction asks for all links.",
        "examples": ["https://example.com", "www.company.co.uk/path"],
        "avoid": ["non-link text"],
    },
    "person_image": {
        "description": "A visible face, portrait, ID photo or photograph containing a person when requested.",
        "examples": ["passport portrait", "profile photo", "person in a screenshot"],
        "avoid": ["logos, statues and illustrations unless the instruction includes them"],
    },
    "custom_request": {
        "description": "A visual or textual target explicitly described by the user's custom instruction.",
        "examples": ["signatures", "company logos", "vehicle registration plates"],
        "avoid": ["anything outside the user's request and the standard privacy categories"],
    },
}

# Compact synthetic examples used as in-context examples. These improve consistency,
# but are not a substitute for a separately run fine-tuning job.
FEW_SHOT_FINDINGS: list[dict[str, object]] = [
    {"input": "Email: alex.taylor@example.com", "category": "email_address", "target": "alex.taylor@example.com"},
    {"input": "Call me on +44 7700 900123", "category": "phone_number", "target": "+44 7700 900123"},
    {"input": "Address: Flat 4, 18 King Street, Coventry CV1 2AB", "category": "home_address", "target": "Flat 4, 18 King Street, Coventry CV1 2AB"},
    {"input": "Applicant: Jordan Lee", "category": "full_name", "target": "Jordan Lee"},
    {"input": "@bluefox_17", "category": "username", "target": "@bluefox_17"},
    {"input": "Account number 12345678", "category": "account_number", "target": "12345678"},
    {"input": "Card 4111 1111 1111 1111", "category": "bank_card_number", "target": "4111 1111 1111 1111"},
    {"input": "FIREWORKS_API_KEY=fw_example_secret", "category": "api_key", "target": "fw_example_secret"},
    {"input": "Authorization: Bearer eyJhbGciOi...", "category": "access_token", "target": "eyJhbGciOi..."},
    {"input": "Password: Winter!2026", "category": "password", "target": "Winter!2026"},
    {"input": "postgresql://admin:secret@db.internal/app", "category": "database_connection_string", "target": "postgresql://admin:secret@db.internal/app"},
    {"input": "-----BEGIN PRIVATE KEY-----", "category": "private_key", "target": "complete key block"},
    {"input": "Server IP 10.0.0.42", "category": "ip_address", "target": "10.0.0.42"},
    {"input": r"C:\Users\Jordan\Documents\private.txt", "category": "file_path", "target": r"C:\Users\Jordan\Documents\private.txt"},
    {"input": "https://example.com/reset?token=secret", "category": "sensitive_url", "target": "complete URL"},
    {"input": "Student ID: 2310456", "category": "student_id", "target": "2310456"},
    {"input": "Employee ID E-10482", "category": "employee_id", "target": "E-10482"},
    {"input": "DOB: 29/06/2007", "category": "date_of_birth", "target": "29/06/2007"},
    {"input": "Your verification code is 381204", "category": "authentication_code", "target": "381204"},
    {"input": "A screenshot containing private message bubbles", "category": "private_chat", "target": "message bubble or panel bbox"},
    {"input": "A visible login QR", "category": "qr_code", "target": "QR bbox"},
]


def compact_playbook() -> str:
    """Return a compact JSON knowledge pack suitable for an inference prompt."""

    compact = {
        category: {
            "description": details["description"],
            "examples": details["examples"],
            "avoid": details["avoid"],
        }
        for category, details in CATEGORY_PLAYBOOK.items()
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def compact_examples() -> str:
    return json.dumps(FEW_SHOT_FINDINGS, ensure_ascii=False, separators=(",", ":"))
