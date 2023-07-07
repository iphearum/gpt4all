# GPT4All REST API
This directory contains the source code to run and build docker images that run a FastAPI app
for serving inference from GPT4All models. The API matches the OpenAI API spec.

## Tutorial

### Starting the app

First build the FastAPI docker image. You only have to do this on initial build or when you add new dependencies to the requirements.txt file:
```bash
python3 -m venv gpt4bin
srource gpt4bin/bin/activate

pip install --upgrade pip
pip install --no-cache-dir -r /requirements.txt

```

Then, start the backend with:

```bash
GPT="gpt4free"
docker image build -t ${GPT}:tag . #5000

docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

PORT=5000
docker run -d -p ${PORT}:5000 ${GPT}:tag

docker compose up --build
```

#### Spinning up your app
Run `docker compose up` to spin up the backend. Monitor the logs for errors in-case you forgot to set an environment variable above.


#### Development
Run

```bash
docker compose up --build
```
and edit files in the `api` directory. The api will hot-reload on changes.

You can run the unit tests with

```bash
make test
```

#### Viewing API documentation

Once the FastAPI ap is started you can access its documentation and test the search endpoint by going to:
```
localhost:80/docs
```

This documentation should match the OpenAI OpenAPI spec located at https://github.com/openai/openai-openapi/blob/master/openapi.yaml


#### Running inference
```python
import openai
openai.api_base = "http://localhost:4891/v1"

openai.api_key = "not needed for a local LLM"


def test_completion():
    model = "gpt4all-j-v1.3-groovy"
    prompt = "Who is Michael Jordan?"
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=50,
        temperature=0.28,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    assert len(response['choices'][0]['text']) > len(prompt)
    print(response)
```
