# Finance Policy Corpus Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `datasets/docs/finance` to six finance policy documents, each at least 1500 characters, so RAG retrieval tests have richer and more ambiguous finance content.

**Architecture:** Keep the RAG implementation unchanged. Use Markdown sample documents as the retrieval corpus and enforce corpus quality through repository sample-data tests.

**Tech Stack:** Markdown, YAML front matter, Python pytest, existing `scripts.ingest_md` parser/discovery helpers.

---

## File Structure

- Modify: `tests/test_ingest_md.py`
  - Update the repository corpus test from 9 total documents to 13 total documents.
  - Add a finance-specific corpus test for six finance categories and minimum document length.
- Modify: `datasets/docs/finance/expense-policy.md`
  - Expand general expense handling to at least 1500 characters.
- Modify: `datasets/docs/finance/travel-policy.md`
  - Expand travel and travel settlement rules to at least 1500 characters.
- Create: `datasets/docs/finance/corporate-card-policy.md`
  - Corporate card issuance, usage, receipt registration, loss, and settlement.
- Create: `datasets/docs/finance/procurement-policy.md`
  - Purchase request, estimates, approvals, ordering, receiving, and exceptions.
- Create: `datasets/docs/finance/vendor-payment-policy.md`
  - Vendor registration, invoices, tax documents, payment schedule, hold reasons.
- Create: `datasets/docs/finance/meal-entertainment-policy.md`
  - Team meals, client entertainment, attendee evidence, limits, and exceptions.
- Modify: `README.md`
  - Update sample corpus description and finance example questions.

## Task 1: Finance Corpus Test

**Files:**
- Modify: `tests/test_ingest_md.py`

- [ ] **Step 1: Update the existing full corpus count**

Change:

```python
assert len(markdown_files) == 9
```

to:

```python
assert len(markdown_files) == 13
```

Add the new finance categories to the expected category set:

```python
"corporate-card",
"procurement",
"vendor-payment",
"meal-entertainment",
```

- [ ] **Step 2: Add failing finance-specific test**

Add:

```python
def test_repository_finance_docs_are_dense_enough_for_retrieval_tests():
    ingest = ingest_module()
    docs_root = Path("datasets/docs/finance")
    markdown_files = ingest.discover_markdown_files(docs_root)

    assert len(markdown_files) == 6

    parsed_docs = [ingest.parse_markdown_file(path) for path in markdown_files]
    metadata_by_category = {metadata["category"]: metadata for metadata, _ in parsed_docs}

    assert set(metadata_by_category) == {
        "expense",
        "travel",
        "corporate-card",
        "procurement",
        "vendor-payment",
        "meal-entertainment",
    }
    assert all(metadata["department"] == "finance" for metadata, _ in parsed_docs)
    assert all(metadata["security_level"] == "internal" for metadata, _ in parsed_docs)

    char_counts = {
        path.name: len(path.read_text(encoding="utf-8"))
        for path in markdown_files
    }
    assert all(count >= 1500 for count in char_counts.values()), char_counts
```

- [ ] **Step 3: Run the new test and verify RED**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_finance_docs_are_dense_enough_for_retrieval_tests -v
```

Expected: FAIL because only two finance files exist and both are below the target length.

## Task 2: Expand Finance Documents

**Files:**
- Modify: `datasets/docs/finance/expense-policy.md`
- Modify: `datasets/docs/finance/travel-policy.md`
- Create: `datasets/docs/finance/corporate-card-policy.md`
- Create: `datasets/docs/finance/procurement-policy.md`
- Create: `datasets/docs/finance/vendor-payment-policy.md`
- Create: `datasets/docs/finance/meal-entertainment-policy.md`

- [ ] **Step 1: Expand or create each finance file**

Ensure each file has valid front matter and at least 1500 characters.

- [ ] **Step 2: Run the finance-specific test and verify GREEN**

Run:

```powershell
docker compose run --rm rag-api pytest tests/test_ingest_md.py::test_repository_finance_docs_are_dense_enough_for_retrieval_tests -v
```

Expected: PASS.

## Task 3: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update sample document list**

Add the four new finance files to the current sample documents list.

- [ ] **Step 2: Update sample corpus wording**

Change the sample corpus description from 9 fictional internal policy documents to 13 fictional internal policy documents, noting that finance now has six documents for retrieval testing.

- [ ] **Step 3: Update finance example questions**

Add examples for corporate card, procurement, vendor payment, and meal/entertainment questions.

## Task 4: Verification

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

- [ ] **Step 3: Run actual ingestion**

Run:

```powershell
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs
```

Expected: `Documents indexed: 13` and nonzero chunk/vector counts.

## Self-Review

- Spec coverage: Covers six finance documents, 1500 character minimum, unique finance categories, and README updates.
- Placeholder scan: No unfinished placeholder markers are used.
- Type consistency: Tests use existing `ingest.discover_markdown_files()` and `ingest.parse_markdown_file()` helpers and existing metadata keys.
