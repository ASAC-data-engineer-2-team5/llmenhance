# Contributing

팀원이 같은 방식으로 작업하기 위한 기본 규칙입니다.

## Commit Message

가능하면 아래 형식을 사용합니다.

```text
type: summary
```

예시:

```text
feat: add login page
fix: handle empty input
docs: update setup guide
```

자주 쓰는 타입:

- `feat`: 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `style`: 포맷팅, 스타일 수정
- `refactor`: 동작 변경 없는 코드 개선
- `test`: 테스트 추가/수정
- `chore`: 설정, 빌드, 기타 작업

## Pull Request

Pull Request에는 아래 내용을 포함합니다.

- 변경 요약
- 테스트 방법
- 스크린샷 또는 실행 결과, UI 변경이 있는 경우
- 리뷰어가 집중해서 봐야 할 부분

## Local Checks

커밋 전 아래 명령을 실행해 기본 품질 검사를 맞춥니다.

```bash
ruff check .
ruff format --check .
pytest
```

처음 클론한 팀원은 한 번만 pre-commit 훅을 설치합니다.

```bash
python -m pip install -r requirements-dev.txt
pre-commit install
```

## Before Merge

- 충돌이 없는지 확인합니다.
- 로컬에서 실행 또는 테스트를 확인합니다.
- 리뷰 코멘트를 반영합니다.
- 불필요한 파일이 포함되지 않았는지 확인합니다.
