# syntax=docker/dockerfile:1
FROM python:3.9-slim

RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install drawnow

COPY allDigCNNMNIST .
COPY SPM_task.py .
COPY main.py .
COPY train_synets.py .
COPY remove_neurons.py .
