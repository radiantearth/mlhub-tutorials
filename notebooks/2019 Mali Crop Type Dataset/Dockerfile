FROM docker.io/tensorflow/tensorflow:latest-gpu-py3

RUN pip install tensorflow
RUN pip install keras==2.6.*
RUN pip install matplotlib
RUN pip install sklearn
COPY data ./data

COPY train.py ./train.py
COPY inference.py ./inference.py
RUN python3 train.py
