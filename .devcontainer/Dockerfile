# Start from a standard Debian image. It's stable and widely used.
FROM debian:bullseye-slim

# Install all the necessary LaTeX packages and common tools.
# 'texlive-full' is very large (~5GB), so we install a smaller set.
# You can add more 'texlive-*' packages here if you need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    latexmk \
    biber \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set a working directory inside the container.
WORKDIR /workspace