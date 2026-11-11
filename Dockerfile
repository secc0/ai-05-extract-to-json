# Build stage: git is needed to install llm-kit from GitHub; it stays out of the final image.
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Runtime stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser
COPY --from=build /venv /venv
WORKDIR /app
COPY examples ./examples
RUN chown appuser /app

USER appuser

# Pass config at runtime and mount your own emails, e.g.:
#   docker run --rm --env-file .env app examples/01.txt
#   docker run --rm --env-file .env -v "$PWD/mail:/app/mail" app mail --batch
ENTRYPOINT ["extract-order"]
CMD ["examples/01.txt"]
