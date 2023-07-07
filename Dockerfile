FROM python:3.8-slim-buster

WORKDIR /

COPY /requirements.txt requirements.txt
# COPY /gpt4all gpt4all
# COPY /models models
# COPY /app.py app.py

RUN python3 -m venv gpt4
RUN . gpt4/bin/activate

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt
# RUN pip install --no-cache-dir -r /requirements.txt

COPY . .

CMD [ "python", "-m" , "app"]
# CMD [ "python", "-m" , "flask", "--app", "interference.app", "run", "--host=0.0.0.0", "--port=5000"]