FROM python:3.9.13

WORKDIR /ai

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN pip install torch==1.9.0 torchvision==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu102

EXPOSE 5555

CMD [ "python", "app.py" ]