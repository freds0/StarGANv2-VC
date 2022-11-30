ARG BASE=nvcr.io/nvidia/pytorch:22.11-py3
FROM ${BASE}

RUN apt-get update -y
RUN pip install -U pip --no-input ; pip install SoundFile torchaudio munch parallel_wavegan torch pydub pyyaml click librosa --no-input