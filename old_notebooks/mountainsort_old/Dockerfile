FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y wget build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/miniconda3/bin:${PATH}"

RUN conda create -n spike_interface_0_97_1 python=3.9 --yes
# activate conda env
SHELL ["conda", "run", "-n", "spike_interface_0_97_1", "/bin/bash", "-c"]
RUN pip install spikeinterface[full,widgets]==0.97.1
RUN pip install --upgrade mountainsort5
RUN conda install -c edeno spectral_connectivity --yes
RUN conda install -c anaconda gitpython -y
RUN conda install -c conda-forge gradio -y
RUN pip install chardet
RUN pip install cchardet

RUN conda init bash
RUN echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

# copy all files in the directory over to container
COPY . .

# CMD ["conda", "run", "-n", "spike_interface_0_97_1", "python", "app.py"]