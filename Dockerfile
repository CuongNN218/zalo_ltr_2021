FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx


RUN mkdir /home/code
COPY . /home/code
WORKDIR /home/code

RUN pip install -r requirements.txt

RUN pip install -e transformers/

RUN mkdir data
RUN mkdir result

ENV data="/data"

RUN chmod +x predict.sh
CMD ./predict.sh

