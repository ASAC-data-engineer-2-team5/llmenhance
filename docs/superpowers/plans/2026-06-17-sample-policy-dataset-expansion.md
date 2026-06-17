# Sample Policy Dataset Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the fictional internal-policy sample dataset so the RAG MVP can exercise realistic HR, finance, security, and general-office questions.

**Architecture:** Keep the existing ingestion architecture unchanged. Add Markdown files with valid YAML front matter under `datasets/docs`, expand the existing leave policy, and add focused tests that verify the sample corpus has the expected metadata coverage and enough content density for meaningful retrieval.

**Tech Stack:** Markdown, YAML front matter, Python pytest, existing `scripts.ingest_md` parser/discovery functions.

---

## File Structure

- Modify: `datasets/docs/hr/leave-policy.md`
  - Expand the existing leave policy with more sections while preserving `category: leave`.
- Create: `datasets/docs/hr/remote-work-policy.md`
  - HR remote-work eligibility, approval, attendance, security, and exceptions.
- Create: `datasets/docs/hr/onboarding-guide.md`
  - New-hire checklist, training, account/device setup, probation check-ins.
- Create: `datasets/docs/finance/expense-policy.md`
  - Expense evidence, deadlines, approval, card usage, rejection reasons.
- Create: `datasets/docs/finance/travel-policy.md`
  - Business trip request, booking, allowances, settlement, changes.
- Create: `datasets/docs/security/privacy-policy.md`
  - Personal-data classification, storage, sharing, retention, incident reporting.
- Create: `datasets/docs/security/device-security.md`
  - Work-device protection, loss reporting, external storage, updates.
- Create: `datasets/docs/general/document-retention.md`
  - Document owners, retention periods, archiving, deletion.
- Create: `datasets/docs/general/meeting-room-policy.md`
  - Meeting-room booking, cancellation, visitors, no-show handling.
- Modify: `tests/test_ingest_md.py`
  - Add a test that parses the repository sample corpus and verifies document count, metadata categories, and body density.
- Modify: `README.md`
  - Update sample-document description and expected ingestion counts from one document to a multi-document corpus.

No changes are planned for `app/chunking.py`, `app/embeddings.py`, `app/vector_store.py`, `app/metadata_store.py`, or `app/rag_pipeline.py`.

## Task 1: Add Corpus Coverage Test

**Files:**
- Modify: `tests/test_ingest_md.py`

- [ ] **Step 1: Add a failing test for repository sample corpus coverage**

Add this test near the existing ingestion tests:

```python
def test_repository_sample_docs_cover_core_policy_topics():
    docs_root = Path("datasets/docs")
    markdown_files = ingest_md.discover_markdown_files(docs_root)

    assert len(markdown_files) == 9

    parsed_docs = [ingest_md.parse_markdown_file(path) for path in markdown_files]
    metadata_by_category = {metadata["category"]: metadata for metadata, _ in parsed_docs}

    assert set(metadata_by_category) == {
        "leave",
        "remote-work",
        "onboarding",
        "expense",
        "travel",
        "privacy",
        "device-security",
        "document-retention",
        "meeting-room",
    }
    assert {metadata["department"] for metadata, _ in parsed_docs} == {
        "hr",
        "finance",
        "security",
        "general",
    }
    assert all(metadata["security_level"] == "internal" for metadata, _ in parsed_docs)
    assert all(len(body.split()) >= 120 for _, body in parsed_docs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_sample_docs_cover_core_policy_topics -v
```

Expected: FAIL because the repository currently contains only `datasets/docs/hr/leave-policy.md`.

## Task 2: Expand HR Sample Documents

**Files:**
- Modify: `datasets/docs/hr/leave-policy.md`
- Create: `datasets/docs/hr/remote-work-policy.md`
- Create: `datasets/docs/hr/onboarding-guide.md`

- [ ] **Step 1: Expand `leave-policy.md`**

Replace the body after existing front matter with sections covering annual leave, half-day leave, hourly leave, sick/family-care exceptions, cancellation, and manager responsibilities.

- [ ] **Step 2: Add `remote-work-policy.md`**

Create a document with this front matter:

```yaml
---
title: 재택근무 운영 규정
doc_type: policy
department: hr
category: remote-work
security_level: internal
---
```

Include sections for eligibility, weekly limit, approval flow, work hours, workplace requirements, information security, and exceptions.

- [ ] **Step 3: Add `onboarding-guide.md`**

Create a document with this front matter:

```yaml
---
title: 신규 입사자 온보딩 안내
doc_type: handbook
department: hr
category: onboarding
security_level: internal
---
```

Include sections for pre-boarding, first-day checklist, account/device setup, mandatory training, mentor check-ins, and probation review.

- [ ] **Step 4: Run the corpus coverage test**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_sample_docs_cover_core_policy_topics -v
```

Expected: FAIL because finance, security, and general documents are not added yet.

## Task 3: Add Finance Sample Documents

**Files:**
- Create: `datasets/docs/finance/expense-policy.md`
- Create: `datasets/docs/finance/travel-policy.md`

- [ ] **Step 1: Add `expense-policy.md`**

Create a document with this front matter:

```yaml
---
title: 경비 처리 규정
doc_type: policy
department: finance
category: expense
security_level: internal
---
```

Include sections for eligible expenses, receipt requirements, settlement deadline, approval flow, corporate-card usage, personal-card reimbursement, and rejection reasons.

- [ ] **Step 2: Add `travel-policy.md`**

Create a document with this front matter:

```yaml
---
title: 출장 및 출장비 정산 규정
doc_type: policy
department: finance
category: travel
security_level: internal
---
```

Include sections for pre-approval, transport and lodging rules, daily allowance, settlement deadline, overseas travel, and itinerary changes.

- [ ] **Step 3: Run the corpus coverage test**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_sample_docs_cover_core_policy_topics -v
```

Expected: FAIL because security and general documents are not added yet.

## Task 4: Add Security Sample Documents

**Files:**
- Create: `datasets/docs/security/privacy-policy.md`
- Create: `datasets/docs/security/device-security.md`

- [ ] **Step 1: Add `privacy-policy.md`**

Create a document with this front matter:

```yaml
---
title: 개인정보 및 민감정보 보호 규정
doc_type: policy
department: security
category: privacy
security_level: internal
---
```

Include sections for personal-data classification, storage locations, sharing restrictions, encryption, retention, deletion, and incident reporting.

- [ ] **Step 2: Add `device-security.md`**

Create a document with this front matter:

```yaml
---
title: 업무 기기 보안 규정
doc_type: policy
department: security
category: device-security
security_level: internal
---
```

Include sections for device assignment, screen lock, OS updates, external storage, loss reporting, return, and remote wipe.

- [ ] **Step 3: Run the corpus coverage test**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_sample_docs_cover_core_policy_topics -v
```

Expected: FAIL because general documents are not added yet.

## Task 5: Add General-Office Sample Documents

**Files:**
- Create: `datasets/docs/general/document-retention.md`
- Create: `datasets/docs/general/meeting-room-policy.md`

- [ ] **Step 1: Add `document-retention.md`**

Create a document with this front matter:

```yaml
---
title: 문서 보존 및 폐기 기준
doc_type: procedure
department: general
category: document-retention
security_level: internal
---
```

Include sections for document owners, retention periods, shared-drive storage, archiving, deletion approval, legal hold, and transfer on role change.

- [ ] **Step 2: Add `meeting-room-policy.md`**

Create a document with this front matter:

```yaml
---
title: 회의실 예약 및 이용 규정
doc_type: procedure
department: general
category: meeting-room
security_level: internal
---
```

Include sections for booking, cancellation, no-show handling, visitor meetings, equipment, food cleanup, and recurring reservations.

- [ ] **Step 3: Run the corpus coverage test**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_sample_docs_cover_core_policy_topics -v
```

Expected: PASS.

## Task 6: Update README Sample Dataset Notes

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the sample document section**

Replace the statement that the current sample document is only HR leave policy with text saying the sample corpus includes HR, finance, security, and general-office documents.

- [ ] **Step 2: Update expected ingestion output**

Change the expected ingestion output from exact one-document counts to example multi-document output. Use wording that chunk count can vary if sample document text changes:

```text
Documents indexed: 9
Chunks created: N
Vectors inserted: N
SQLite rows inserted: 9 + N
```

- [ ] **Step 3: Update example questions if needed**

Keep the existing leave question and add examples for remote work, expenses, travel, privacy, device security, onboarding, and meeting rooms.

## Task 7: Full Verification

**Files:**
- No file changes.

- [ ] **Step 1: Run focused ingestion tests**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
docker compose run --rm rag-api pytest -v
```

Expected: PASS.

- [ ] **Step 3: Verify Qdrant health**

Run:

```powershell
curl http://localhost:6333
```

Expected: JSON response from Qdrant.

- [ ] **Step 4: Verify app healthcheck**

Run:

```powershell
docker compose run --rm rag-api python -m app.healthcheck
```

Expected: Settings print successfully, including Ollama, Qdrant, model, chunk, and retrieval settings.

## Self-Review

- Spec coverage: The plan covers the approved 9-document corpus, metadata fields, content density, retrieval question coverage, fallback preservation, and no architecture changes.
- Placeholder scan: The plan intentionally avoids TBD/TODO placeholders and names every file and command.
- Type consistency: The test uses existing `ingest_md.discover_markdown_files()` and `ingest_md.parse_markdown_file()` functions, and uses existing metadata field names exactly.

