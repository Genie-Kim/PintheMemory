FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo  \
        cmake protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} pinmem -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER pinmem
WORKDIR /home/pinmem

ENV PATH="/home/pinmem/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
        python3 get-pip.py --user && \
        rm get-pip.py

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --user tqdm scikit-image opencv-python Pillow tensorflow tensorboardX thop kmeans1d

RUN git config --global alias.lg "log --all --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr)%C(bold blue)<%an>%Creset' --abbrev-commit"
RUN git config --global alias.s "status -s"

RUN sudo apt-get install -y tmux tree
RUN pip3 install scipy==1.1.0
RUN pip3 install scikit-image==0.16.2

