FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG CODSPEED_VERSION=5.1.0
ENV DEBIAN_FRONTEND=noninteractive
ENV CODSPEED_PROFILER_ENABLED=false
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git python3 && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://codspeed.io/v${CODSPEED_VERSION}/install.sh" -o /tmp/install-codspeed.sh && \
    sh /tmp/install-codspeed.sh && \
    codspeed --version

ENV CODSPEED_ISOLATION=false
WORKDIR /workspace
