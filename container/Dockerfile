FROM pytorch/pytorch

# Install some basic apt packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    sudo vim nano git curl wget mc less \
    openssh-server libssl-dev \
    ffmpeg libsm6 libxext6  && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN echo "%docker  ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/inside_sudoers

RUN pip install tensorboard ssl_eval numpy pkbar

ENV PYTHONPATH=/workspace
WORKDIR /workspace

COPY entry.sh /entry.sh
RUN chmod +x /entry.sh
ENTRYPOINT ["/entry.sh"]
