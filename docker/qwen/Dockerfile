# Optional, reproducible Linux environment for the NOVA Qwen consumer and its tests.
# The consumer itself needs only Python; Docker is not required (see docs/qwen_integration.md).
# Build from the repository root:
#   docker build -f docker/qwen/Dockerfile -t nova-qwen:0.1 .
FROM python:3.12-slim@sha256:5b8b1421ccb6a525860a77863c049eab5847076cda34e030c75a80c58faf3f3c

ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY requirements-qwen.txt requirements-qwen-dev.txt /tmp/req/
RUN python -m pip install -r /tmp/req/requirements-qwen-dev.txt
# The working directory and outputs/ must belong to the non-root user (WORKDIR alone would create them as root).
RUN useradd --create-home --uid 10001 nova && mkdir -p /nova/outputs && chown -R nova:nova /nova
WORKDIR /nova
COPY --chown=nova:nova . /nova
USER nova
# Default: the offline suite. Live: see docker/qwen/compose.yaml.
CMD ["python", "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/qwen"]
