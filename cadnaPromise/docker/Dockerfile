FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. install dependencies + gcc/g++-13
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    gcc-13 \
    g++-13 \
    git \
    rsync \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# 2. python -> python3
# ------------------------------------------------------------
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------
# 3. upgrade pip + test tools
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel pytest

# ------------------------------------------------------------
# 4. copy repo，and install promise from source
#    (requires docker build context to be repo_root)
# ------------------------------------------------------------
WORKDIR /opt/cadnaPromise
COPY . .

RUN pip install -e .

# Optional: build-time verification
RUN promise --version

# ------------------------------------------------------------
# 5. entrypoint: auto activate-promise (using g++-13)
# ------------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
