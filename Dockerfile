FROM python:3.9.13

WORKDIR /ai

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

EXPOSE 5000

CMD [ "python", "app.py" ]