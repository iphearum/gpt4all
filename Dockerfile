FROM python:3.10-slim-buster

# RUN apt-get update && apt-get install -y python3-pip

WORKDIR /

COPY /requirements.txt requirements.txt

RUN python3 -m venv gpt4bin
RUN . gpt4bin/bin/activate

RUN pip install --upgrade pip

# RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install -r /requirements.txt

COPY app/ .

# COPY ../models/gpt4all/ggml-gpt4all-j-v1.3-groovy.bin models/


CMD [ "python", "-m" , "app"]
# CMD [ "python", "-m" , "flask", "--app", "interference.app", "run", "--host=0.0.0.0", "--port=5000"]