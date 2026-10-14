FROM mambaorg/micromamba:2.0.5@sha256:8d0d4cad074afe3ac93b58dce88bd2659153d3fa6de3a6968f501710e8d3c832 AS micromamba

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV TZ=Etc/UTC

# guestfish runs its own appliance under QEMU TCG. Its kernel is part of this
# image, not the server's /boot, and no loop/FUSE/KVM device is passed through.
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl git rsync openssh-client openssh-server pciutils sudo screen \
      build-essential bc bison flex cpio file patch unzip zip wget \
      autoconf automake libtool gawk texinfo help2man bzip2 xz-utils zstd time \
      python3 python3-yaml pkg-config libssl-dev libelf-dev \
      zlib1g-dev libgmp10 libmpfr6 libmpc3 libncurses5 libncurses-dev libtinfo5 \
      device-tree-compiler dosfstools e2fsprogs util-linux kmod \
      libguestfs-tools qemu-system-x86 qemu-utils supermin \
      linux-image-generic locales && \
    locale-gen en_US.UTF-8 && \
    mkdir -p /opt/guestfs-appliance && \
    libguestfs-make-fixed-appliance /opt/guestfs-appliance && \
    chmod -R a+rX /opt/guestfs-appliance && \
    dpkg-query -W > /opt/stage2-os-packages.tsv && \
    rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* && \
    mkdir -p /run/sshd /root/.ssh && chmod 0700 /root/.ssh && usermod -p '*' root

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    CHIPYARD_DIR=/opt/chipyard \
    PYTORCH_CHIPYARD_STAGE2_CONTAINER=1 STAGE2_EXECUTION=local \
    LIBGUESTFS_BACKEND=direct LIBGUESTFS_BACKEND_SETTINGS=force_tcg \
    LIBGUESTFS_PATH=/opt/guestfs-appliance FORCE_UNSAFE_CONFIGURE=1
ENV PATH=/opt/chipyard/.conda-env/bin:/opt/chipyard/.conda-env/riscv-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
COPY docker/stage2/conda-linux-64.lock /opt/stage2-support/
RUN micromamba create -y -p /opt/chipyard/.conda-env -f /opt/stage2-support/conda-linux-64.lock && \
    micromamba clean --all --yes

COPY docker/stage2/sources.lock docker/stage2/install-sources.sh /opt/stage2-support/
COPY docker/stage2/patches/ /opt/stage2-support/patches/
RUN bash /opt/stage2-support/install-sources.sh

COPY docker/stage2/requirements.txt /opt/stage2-support/
RUN /opt/chipyard/.conda-env/bin/python -m pip install --no-cache-dir --no-deps -r /opt/stage2-support/requirements.txt

COPY scripts/ /opt/pytorch-chipyard/scripts/
COPY docker/stage2/ /opt/stage2-support/
COPY bitstreams/ /opt/firesim-artifacts/
COPY docker/stage2-entrypoint.sh /usr/local/bin/pytorch-chipyard-stage2
COPY docker/stage2-local-entrypoint.sh /usr/local/bin/pytorch-chipyard-stage2-local
RUN /usr/bin/python3 /opt/stage2-support/finalize-image.py && \
    /usr/bin/python3 /opt/stage2-support/patch-runtime.py /opt/chipyard && \
    /usr/bin/python3 /opt/stage2-support/patch-local.py /opt/chipyard && \
    chmod 0755 /usr/local/bin/pytorch-chipyard-stage2* /usr/local/bin/firesim-* && \
    git config --system --add safe.directory '*' && \
    /usr/bin/python3 /opt/pytorch-chipyard/scripts/prepare-firesim-hwdb.py \
      --bitstream-dir /opt/firesim-artifacts --output /opt/stage2-metadata/config_hwdb.yaml

LABEL org.opencontainers.image.title="PyTorch-Chipyard Stage2" \
      org.pytorch-chipyard.stage2.execution="full-container-u250-v1" \
      org.pytorch-chipyard.stage2.programmer="host-vivado-ssh-v1" \
      org.pytorch-chipyard.stage2.validation="hardware-pending" \
      org.pytorch-chipyard.stage2.table4="excluded"
WORKDIR /opt/pytorch-chipyard
ENTRYPOINT ["/usr/local/bin/pytorch-chipyard-stage2-local"]
CMD ["--help"]
