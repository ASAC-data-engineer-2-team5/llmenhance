# Team Workflow

## Daily Flow

1. 이슈 또는 작업 항목을 확인합니다.
2. `main` 브랜치를 최신 상태로 맞춥니다.
3. 작업 브랜치를 생성합니다.
4. 기능을 구현하고 커밋합니다.
5. Pull Request를 열고 리뷰를 요청합니다.
6. 리뷰 반영 후 `main`에 병합합니다.

## Recommended Commands

```bash
git switch main
git pull origin main
git switch -c feature/my-task
```

작업 후:

```bash
git status
git add .
git commit -m "feat: describe change"
git push -u origin feature/my-task
```

## Issue Labels

- `feature`: 새 기능
- `bug`: 버그
- `docs`: 문서
- `design`: 화면 또는 UI
- `question`: 논의 필요

## Review Checklist

- 요구사항에 맞게 동작하는가?
- 실행 또는 테스트 방법이 적혀 있는가?
- 불필요한 파일이 포함되지 않았는가?
- 팀원이 이해할 수 있는 이름과 구조인가?
