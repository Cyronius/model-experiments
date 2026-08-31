# Standalone Railway deploy for titler's demo server (examples/web/server.py):
# FastAPI + ONNX Runtime, CPU-only. See .claude/plans/deploy-titler-railway.md.
FROM python:3.11-slim
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
# CPU-only torch wheel first — PyPI's default linux wheel drags in CUDA
# (~2GB+ extra) that this CPU-only deploy never uses.
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system -e .

COPY scripts/common ./scripts/common
COPY examples/web ./examples/web
COPY model/titler-v15-scratch-web ./model/titler-v15-scratch-web

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Railway injects PORT; --host 0.0.0.0 is required in a container (the
# script's own default, 127.0.0.1, is unreachable from outside it).
ENTRYPOINT ["/bin/sh", "-c", "exec python examples/web/server.py --model model/titler-v15-scratch-web --host 0.0.0.0 --port ${PORT:-8000}"]
