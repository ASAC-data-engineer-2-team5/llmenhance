# MVP 범위

## 프로젝트 배경

`planet_team05`는 임직원 420명 규모의 가상 B2B 데이터/AI SaaS 회사이다. 회사의 사내 규정은 인사, 휴가, 재택근무, 정보보안, 재무, 자산관리, 사내 위키형 운영 가이드 등 여러 문서와 시스템에 흩어져 있다.

직원들은 규정 조항 번호를 직접 묻기보다는 다음과 같은 상황형 질문을 자주 한다.

- 재택근무일에 카페에서 일해도 되나요?
- 병원 진료로 2시간 자리를 비우면 외출, 병가, 반차 중 무엇으로 처리되나요?
- 퇴근 기록을 깜빡했는데 언제까지 정정해야 하나요?
- 재택근무 중 장애 대응으로 밤 10시 이후까지 일하면 초과근무로 인정되나요?

본 MVP는 운영 문서 포맷 파서, 유료 모델 비교, MCP 도구, 웹 UI를 붙이기 전에 가장 작은 단위의 온프레미스 사내 규정 RAG 흐름을 검증하는 것을 목표로 한다.

## MVP 목표

MVP에서는 다음 흐름이 로컬에서 동작하는지 확인한다.

1. Markdown으로 작성된 가상 사내 규정 문서를 읽는다.
2. 각 문서를 공통 정규화 문서 모델로 변환한다.
3. 한국어 규정 문서의 장, 절, 조, 항 구조를 고려해 chunk를 만든다.
4. 사용자의 질문과 관련 있는 chunk를 검색한다.
5. 답변 생성에 사용할 수 있는 근거 정보를 문서 ID, 문서명, 제목, 원본 경로와 함께 반환한다.

이 단계에서는 LLM 답변 생성보다 `검색 근거를 안정적으로 찾는 것`을 먼저 검증한다. 검색 품질을 측정할 수 있어야 이후 로컬 LLM, 유료 LLM, reranker, MCP 도구를 붙였을 때 개선 여부를 비교할 수 있다.

## MVP 포함 범위

- `data/policies/markdown/` 아래의 Markdown 사내 규정 문서
- 이후 다른 loader가 재사용할 수 있는 공통 `NormalizedDocument` 모델
- 한국어 규정 문서의 조항과 제목을 고려한 chunking
- 로컬 기준선 실험을 위한 단순 lexical retriever
- `data/policies/normalized/`와 `data/policies/chunks/` 아래 JSON/JSONL 출력
- `data/eval/` 아래의 작은 평가 질문셋
- `src/llmenhance/mvp/` 아래 CLI 형태의 실행 스크립트

## MVP 제외 범위

- PDF, DOCX, HWP, HWPX 파서
- Confluence, Notion, SharePoint, GitBook, HTML 사내 위키 connector
- Vector DB 연동
- embedding 모델 선정
- 로컬 LLM serving
- 유료 LLM 비교
- MCP 도구, agent workflow, skills harness
- 인증, 권한관리, 실제 임직원 데이터 처리
- 웹 UI

## 성공 기준

- Markdown 정책 문서를 수작업 복사 없이 pipeline으로 ingest할 수 있다.
- 각 chunk가 `chunk_id`, `document_id`, `title`, `heading`, `source_path`, `text` 메타데이터를 가진다.
- 한국어 직원 질문을 넣었을 때 관련 chunk가 1개 이상 검색된다.
- 나중에 `PdfLoader`, `DocxLoader`, `HwpxLoader`, `HwpLoader`, 사내 위키 loader를 추가하더라도 chunker와 retriever 인터페이스를 크게 바꾸지 않는다.

## 설계 원칙

RAG pipeline이 Markdown 파일 형식에 직접 의존하면 안 된다. Markdown은 첫 번째 입력 포맷일 뿐이며, 이후 모든 단계는 공통 정규화 문서 모델을 입력으로 받아야 한다.

