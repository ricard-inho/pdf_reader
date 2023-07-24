FROM nvidia/cudagl:11.4.2-devel-ubuntu18.04

RUN apt-get update -q && \
    apt-get install -y -qq  \
    curl \
    vim \
    zip \
    unzip \
    cmake \
    ffmpeg \
    wget \
    libnss3-tools 


# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version


ARG env_name=pdf_reader

# Conda environment
RUN conda create -n ${env_name} python=3.8 -y

RUN /bin/bash -c ". activate ${env_name}; conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y"
RUN /bin/bash -c ". activate ${env_name}; pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub"
RUN /bin/bash -c ". activate ${env_name}; pip install InstructorEmbedding sentence_transformers"
RUN /bin/bash -c ". activate ${env_name}; pip install tiktoken"

# Setup conda env
RUN conda init \
	&& echo "conda activate ${env_name}" >> ~/.bashrc

WORKDIR /