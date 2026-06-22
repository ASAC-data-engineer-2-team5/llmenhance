# MVP 로드맵

## Phase 0: 기획 및 데이터셋 형태 정의

- `planet_team05`용 가상 사내 규정 문서를 만든다.
- Markdown을 초기 작성 포맷으로 사용한다.
- 공통 정규화 문서 모델을 정의한다.
- MVP 입력/출력 디렉터리를 정한다.

## Phase 1: Markdown 기반 MVP

- `MarkdownLoader`를 구현한다.
- `PolicyNormalizer`를 구현한다.
- `PolicyChunker`를 구현한다.
- 정규화된 문서를 JSON으로 export한다.
- chunk 결과를 JSONL로 export한다.
- 단순 lexical retriever를 구현한다.
- `agent.md` 기준으로 MVP agent harness를 정의한다.
- 상위 근거 chunk를 반환하는 최소 `ask` 스크립트를 추가한다.
- 평가 질문셋 기준 Recall@k를 계산하는 `evaluate` 스크립트를 추가한다.

## Phase 2: 평가 기준선 구축

- 직원이 실제로 물어볼 법한 질문을 최소 30개 만든다.
- 각 질문별 기대 문서와 기대 heading을 라벨링한다.
- Recall@3, Recall@5 같은 단순 지표로 검색 품질을 측정한다.
- 여러 문서를 함께 봐야 하는 질문을 따로 표시한다.

## Phase 3: 로컬 RAG 프로토타입

- embedding 모델을 추가한다.
- vector DB를 추가한다.
- lexical, vector, hybrid retrieval을 비교한다.
- 근거 조항을 인용하는 local LLM 답변 생성을 붙인다.
- 근거가 약할 때 "규정에서 확인할 수 없습니다"라고 답하는 동작을 추가한다.

## Phase 4: 실제 문서 포맷 확장

- Markdown 정책 문서를 PDF와 DOCX test fixture로 변환한다.
- `PdfLoader`와 `DocxLoader`를 추가한다.
- 한국 기업 문서 현실성을 위해 `HwpxLoader`를 추가한다.
- 구형 `.hwp`는 best-effort 직접 추출 또는 HWPX/PDF 변환 기반 경로로 처리한다.

## Phase 5: 사내 위키 및 connector 확장

- 먼저 정적 HTML 위키 fixture를 추가한다.
- Confluence, Notion, SharePoint, GitBook connector 인터페이스를 설계한다.
- source URL, 마지막 수정 시각, 접근 권한 메타데이터를 보존한다.
- 공식 규정 문서와 위키 운영 가이드가 충돌할 때의 우선순위를 정의한다.

## Phase 6: 모델 및 harness 비교

- local LLM, paid LLM, retrieval-only baseline을 비교한다.
- reranker 실험을 추가한다.
- prompt template과 guardrail을 추가한다.
- `agent.md` harness 기준으로 답변 품질, 출처 표시, 확인 불가 처리를 비교한다.
- 검색 및 citation 품질을 측정할 수 있게 된 뒤 MCP 또는 skill-like tool을 붙인다.

## 추천 팀 역할 분담

- 데이터/평가 담당: 정책 문서, 질문셋, golden evidence 라벨 관리
- ingestion 담당: loader, normalizer, chunker, metadata 관리
- 검색/모델 담당: lexical retrieval, embedding, vector DB, reranker 실험
- 앱/DevOps 담당: CLI, API, UI, Docker, CI, 문서화
