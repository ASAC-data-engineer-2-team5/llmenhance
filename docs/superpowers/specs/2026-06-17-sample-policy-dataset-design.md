# Sample Policy Dataset Expansion Design

## Goal

Expand the MVP sample documents so the local RAG pipeline can be tested with practical company-policy questions rather than a single leave-policy smoke test.

The dataset should remain fictional, compact, and easy to ingest locally. It should cover enough HR, finance, security, and general-office topics to exercise metadata hard filtering, vector retrieval, grounded answering, source output, and fallback behavior.

## Document Set

The MVP dataset will contain these Markdown documents under `datasets/docs`:

```text
hr/leave-policy.md
hr/remote-work-policy.md
hr/onboarding-guide.md
finance/expense-policy.md
finance/travel-policy.md
security/privacy-policy.md
security/device-security.md
general/document-retention.md
general/meeting-room-policy.md
```

The existing `hr/leave-policy.md` will be expanded instead of replaced with unrelated content.

## Metadata

Each document will include YAML front matter with the existing ingestion metadata fields:

```yaml
title: ...
doc_type: policy | procedure | handbook
department: hr | finance | security | general
category: leave | remote-work | onboarding | expense | travel | privacy | device-security | document-retention | meeting-room
security_level: internal
```

These values are chosen to match the current SQLite hard-filter model. No natural-language-to-SQL or automatic metadata inference will be added.

## Content Shape

Each document will have 4-7 short sections. The content should be realistic enough for retrieval tests but not so long that ingestion becomes slow for the MVP.

Policy text should include concrete facts that users can ask about:

- deadlines, such as 3 business days before leave or 5 business days after travel
- approval owners, such as team leads, finance, security, or HR
- required evidence, such as receipts, approvals, incident reports, and training completion
- allowed and disallowed behavior
- exception handling

The documents will not include real company secrets, real personal data, real vendor contracts, or legal advice.

## Retrieval Coverage

The expanded set should support meaningful answers to questions such as:

```text
연차 신청은 며칠 전까지 해야 하나요?
재택근무는 주 몇 회까지 가능한가요?
재택근무 승인 절차는 어떻게 되나요?
출장비 정산은 언제까지 해야 하나요?
경비 처리 시 어떤 증빙이 필요한가요?
개인정보가 포함된 문서는 어떻게 보관해야 하나요?
노트북을 분실하면 누구에게 신고해야 하나요?
신규 입사자는 언제까지 보안 교육을 완료해야 하나요?
회의실 예약을 취소하지 않으면 어떻게 되나요?
```

The dataset should also leave some plausible questions unanswered so fallback behavior can still be tested. For example, questions about 육아휴직 급여, 해외 주재원 수당, or stock option rules should not be answered unless a matching document is later added.

## Testing Strategy

Because this change primarily expands sample data, implementation should focus on:

- preserving valid Markdown and YAML front matter
- verifying ingestion still discovers all documents
- updating sample-count expectations where documentation or tests rely on the old one-document dataset
- running the existing ingestion and RAG tests

No changes are planned for chunking, embedding, SQLite schema, Qdrant schema, or Qwen prompt behavior.

