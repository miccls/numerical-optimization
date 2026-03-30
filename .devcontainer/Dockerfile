FROM ext-dockerio-remote.artifactory.trioptima.net/debian:stable-slim

# Based on https://stash.trioptima.net/projects/RED/repos/tribalance/browse/docker/trireduce-base/Dockerfile

ARG DEBIAN_FRONTEND=noninteractive

# Basic setup to work within trioptima firewall and setting up locale

ADD https://artifactory.trioptima.net/artifactory/certs/trioptima-root-ca-local.crt /usr/local/share/ca-certificates/

# Need to run this step before changing the apt/sources. Trying to install ca-certificates with our own sources fails on SSL cert error
RUN <<EOF
set -eux

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \

chmod a+rx -R /usr/local/share/ca-certificates
update-ca-certificates --fresh
EOF

RUN <<EOF
set -eux

rm -rf /etc/apt/sources.list.d || true
echo "deb https://artifactory.trioptima.net/artifactory/ext-debian-remote stable main" > /etc/apt/sources.list
echo "deb https://artifactory.trioptima.net/artifactory/ext-debian-security-remote stable-security main" >> /etc/apt/sources.list
echo "deb https://artifactory.trioptima.net/artifactory/ext-debian-remote stable-updates main" >> /etc/apt/sources.list

apt-get update
apt-get install -y --no-install-recommends \
  locales

echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen
EOF

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8

# Install stuff we need for development, LaTeX compilers etc
RUN <<EOF
set -eux

apt-get update
apt-get install -y --no-install-recommends \
    sudo \
    git \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-extra \
    texlive-bibtex-extra \
    latexmk \
    biber \
    texlive-extra-utils \
    libyaml-tiny-perl \
    libfile-homedir-perl \
    libunicode-linebreak-perl

# Clean up apt cache
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=bookclubber
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# To make node in vscode use the trioptima certs
ENV NODE_OPTIONS="--use-openssl-ca"

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Set a working directory inside the container.
WORKDIR /workspace