# TensorGuard — reproducible container image (Step 71).
#
# Multi-stage build: a builder stage produces a wheel, the final stage installs
# only that wheel into a slim runtime so the image carries no build toolchain.
# The image entrypoint is the `tensorguard` console script, so
#   docker run --rm -v "$PWD:/work" ghcr.io/thehalleyyoung/tensorguard verify model.py
# behaves exactly like a local install.

FROM python:3.12-slim AS builder

WORKDIR /build

# Build deps only; reproducible wheels honor SOURCE_DATE_EPOCH.
RUN pip install --no-cache-dir build

# Copy the minimum needed to build the wheel (see .dockerignore for exclusions).
COPY pyproject.toml MANIFEST.in LICENSE README.md ./
COPY src ./src

ARG SOURCE_DATE_EPOCH=1700000000
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
RUN python -m build --wheel --no-isolation --outdir /dist


FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="TensorGuard" \
      org.opencontainers.image.description="Sound static shape/device/phase/dtype/gradient verification for PyTorch nn.Module" \
      org.opencontainers.image.source="https://github.com/thehalleyyoung/tensorguard" \
      org.opencontainers.image.licenses="MIT"

# Install the built wheel and its pinned solver dependency.
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Run as a non-root user; mount the project to analyze at /work.
RUN useradd --create-home --uid 1000 tg
USER tg
WORKDIR /work

ENTRYPOINT ["tensorguard"]
CMD ["--help"]
