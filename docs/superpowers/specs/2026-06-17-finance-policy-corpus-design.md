# Finance Policy Corpus Expansion Design

## Goal

Expand the finance sample corpus so the RAG MVP can test retrieval quality with multiple similar but distinct finance policies.

The current finance corpus has only two short files, `expense-policy.md` and `travel-policy.md`. That is enough for a demo answer, but not enough to evaluate semantic retrieval, metadata filtering, source selection, or confusion between overlapping finance terms such as expense, reimbursement, card, settlement, approval, vendor, invoice, and meal.

## Scope

This change applies only to `datasets/docs/finance`.

The finance corpus will contain six Markdown documents:

```text
finance/expense-policy.md
finance/travel-policy.md
finance/corporate-card-policy.md
finance/procurement-policy.md
finance/vendor-payment-policy.md
finance/meal-entertainment-policy.md
```

The two existing files will be expanded. Four new files will be added.

## Length And Density

Each finance document should be at least 1500 characters including front matter. The goal is not to make long filler text, but to include enough sections and overlapping terminology for retrieval tests.

Each document should contain:

- valid YAML front matter
- 6-8 policy sections
- concrete deadlines, approval owners, evidence requirements, limits, exceptions, and rejection cases
- enough similar language to create realistic retrieval ambiguity

## Metadata

Each document will use the existing metadata model:

```yaml
doc_type: policy | procedure
department: finance
category: expense | travel | corporate-card | procurement | vendor-payment | meal-entertainment
security_level: internal
```

No schema change, natural-language-to-SQL, or metadata inference will be added.

## Retrieval Coverage

The expanded corpus should support questions such as:

```text
경비 처리 시 어떤 증빙이 필요한가요?
개인카드 경비는 언제까지 제출해야 하나요?
출장비 정산은 언제까지 해야 하나요?
법인카드를 분실하면 어떻게 해야 하나요?
구매 요청은 언제 견적을 받아야 하나요?
업체 대금 지급일은 언제인가요?
회식비와 접대비는 어떤 기준으로 처리하나요?
```

It should also make confusing questions useful for testing. For example, "카드 영수증은 언제 등록해야 하나요?" may retrieve corporate-card or expense content depending on metadata filters and wording.

## Testing Strategy

Add or update tests so the repository sample corpus verifies:

- total sample corpus still has expected categories
- finance corpus has exactly six documents
- every finance document has at least 1500 characters
- all finance documents use `department: finance`
- finance categories are unique and match the expected set

Update README sample dataset notes and expected ingestion output to mention the larger finance corpus. Existing RAG implementation behavior should remain unchanged.

