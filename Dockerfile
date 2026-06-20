FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# vertexai 모듈 생성
RUN python -c "\
import os, langchain_community; \
path = os.path.join(os.path.dirname(langchain_community.__file__), 'chat_models'); \
open(os.path.join(path, 'vertexai.py'), 'w').write('class ChatVertexAI: pass\n')"


COPY . /app

CMD ["python", "-m", "app.healthcheck"]
