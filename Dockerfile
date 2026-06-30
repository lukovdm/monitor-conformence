# uv binary is copied from the official image (pinned to match the
# uv_build backend required in pyproject.toml).
FROM ghcr.io/astral-sh/uv:0.11.4 AS uv

FROM lukovdm/paynt:tover

RUN apt-get update && apt-get install -y \
    texlive \
    texlive-xetex \
    texlive-science \
    pkg-config \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev

# Install uv (not present in the base image).
COPY --from=uv /uv /uvx /bin/

# Reuse the virtual environment from the base image (it already provides
# stormpy and paynt, which are not on PyPI / not in uv.lock).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy

# Copy the project files into the container
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock
COPY ./README.md /app/README.md
WORKDIR /app

RUN uv sync --frozen --inexact

COPY . /app
