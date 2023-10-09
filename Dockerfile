FROM python:3.11.5-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt 

COPY . .

EXPOSE 2000

CMD ["python", "main.py"]
