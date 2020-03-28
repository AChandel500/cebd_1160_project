FROM ubuntu:18.04

WORKDIR /app

COPY Median_Value_Prediction.py /app
COPY data /app/data
COPY requirements.txt /app

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "./Median_Value_Prediction.py"]
