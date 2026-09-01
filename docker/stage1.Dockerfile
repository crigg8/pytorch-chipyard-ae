FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_VERSION=24.11.3
ARG MINICONDA_INSTALLER_URL=https://repo.anaconda.com/miniconda/Miniconda3-py312_24.11.1-0-Linux-x86_64.sh
# Triton's setup.py otherwise defaults to 2 * os.cpu_count().  On the review
# server that becomes -j160 and can exhaust memory during the C++ build.
ARG MAX_JOBS=16
ARG TRITON_PARALLEL_LINK_JOBS=2

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      bzip2 \
      ca-certificates \
      curl \
      file \
      git \
      make \
      openssh-client \
      patch \
      rsync \
      xz-utils \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "${MINICONDA_INSTALLER_URL}" -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/anaconda3 && \
    rm -f /tmp/miniconda.sh

ENV PATH=/root/anaconda3/envs/pytorch-chipyard/bin:/root/anaconda3/bin:${PATH}

RUN printf '%s\n' \
      'channels:' \
      '  - conda-forge' \
      'channel_priority: strict' \
      'always_yes: true' \
      'changeps1: false' \
      > /root/.condarc && \
    conda install --override-channels -c conda-forge -n base conda="${CONDA_VERSION}" -y && \
    conda clean -afy && \
    conda --version | grep -F "conda ${CONDA_VERSION}"

WORKDIR /opt/pytorch-chipyard

COPY . .

RUN test -f pytorch/torch/_inductor/__init__.py || \
      (echo "missing initialized submodule: pytorch" >&2; exit 1) && \
    test -f triton/python/requirements.txt || \
      (echo "missing initialized submodule: triton" >&2; exit 1) && \
    test -f triton_chipyard/CMakeLists.txt || \
      (echo "missing initialized submodule: triton_chipyard" >&2; exit 1) && \
    test -f llvm-project/llvm/CMakeLists.txt || \
      (echo "missing initialized submodule: llvm-project" >&2; exit 1) && \
    test -f buddy-mlir/CMakeLists.txt || \
      (echo "missing initialized submodule: buddy-mlir" >&2; exit 1)

ENV CONDA_ENV_NAME=pytorch-chipyard
ENV PYTORCH_CHIPYARD_CONDA_ENV=pytorch-chipyard

RUN MAX_JOBS="${MAX_JOBS}" \
    TRITON_PARALLEL_LINK_JOBS="${TRITON_PARALLEL_LINK_JOBS}" \
    bash scripts/install.sh && \
    conda clean -afy && \
    rm -rf /root/.cache/pip /tmp/triton-chipyard-cache

RUN install -m 0755 docker/stage1-entrypoint.sh /usr/local/bin/pytorch-chipyard-stage1

ENTRYPOINT ["/usr/local/bin/pytorch-chipyard-stage1"]
CMD ["bash"]
