FROM nvcr.io/nvidia/pytorch:21.09-py3

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

RUN pip install tqdm dotmap matplotlib opencv-python pandas numpy jupyter sklearn
RUN pip install tensorboard

## suncet python dependencies
RUN pip install PyYaml numpy submitit
RUN pip install pudb

# Apex
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..

### workaround for distributed count_all bug see: https://github.com/NVIDIA/apex/pull/1150
RUN cd /opt/conda/lib/python3.8/site-packages/apex/parallel && \
    curl -O https://raw.githubusercontent.com/NVIDIA/apex/8813adc6a2af04fb27fac3d0c9825ab71a565b4a/apex/parallel/optimized_sync_batchnorm_kernel.py && \
    cd

### Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3.tar.gz &&  \
    tar -zxvf cmake-3.21.3.tar.gz && rm cmake-3.21.3.tar.gz &&  \
    cd cmake-3.21.3 && ./bootstrap && make && make install

# Install pykeops
# RUN git clone --recursive https://github.com/getkeops/keops.git@v1.5
RUN git clone --recursive https://github.com/getkeops/keops.git /libkeops
RUN sed -i '10,13s/./#&/' /libkeops/pykeops/torch/__init__.py
RUN mkdir /libkeops/pykeops/build && chmod 777 /libkeops/pykeops/build  # TODO: It is may not necessary
RUN cd /libkeops/keops/build && cmake .. && cd /libkeops/keops/build && make VERBOSE=0 && cd # TODO: It is may not necessary

RUN ln -s /libkeops/pykeops /opt/conda/lib/python3.8/site-packages/pykeops

ENV PYTHONPATH=/workspace
WORKDIR /workspace

COPY entry.sh /entry.sh
RUN chmod +x /entry.sh
ENTRYPOINT ["/entry.sh"]
