FROM python:3.11.7

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

EXPOSE 2000

CMD ["python", "main.py"]
