import os
import time
import json
import random

from gpt4all import GPT4All
from flask import Flask, request, Response
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

models = [
    # finished
    # {"filename": "ggml-mpt-7b-instruct.bin"},
    # {"filename": "ggml-v3-13b-hermes-q5_1.bin"},
    # {"filename": "ggml-gpt4all-j-v1.3-groovy.bin"},

    # needed for download
    {"filename": "ggml-gpt4all-l13b-snoozy.bin"},
    # {"filename": "ggml-mpt-7b-chat.bin"},
]
MODELPATH = "../models/gpt4all/"
for i in models:
    llm = GPT4All(i['filename'],MODELPATH)

public_key = "pJNAtlAqCHbUDTrDudubjSKeUVgbOMvkRQWMLtscqsdiKmhI"

@app.route("/chat/completions", methods=['POST'])
def chat_completions():
    streaming = request.json.get('stream', False)
    model = request.json.get('model', 'gpt-3.5-turbo')
    messages = request.json.get('messages')
    top_k = request.json.get('top_k',1024)
    barer = request.headers.get('Authorization')
        
    if barer is None:
        barer = 'unknown'
    else:
        barer = barer.strip().split(" ")[1] if len(barer.strip().split(" ")) > 1 else 'unknown'

    if barer != f"pk-{public_key}":
        return Response('Unauthorized', status=401)
    
    # response = llm.generate(messages=messages, max_tokens=10, top_k=top_k, streaming=streaming)
    if not streaming:
        response = llm.chat_completion(messages, verbose=False, streaming=streaming)
        response['model'] = 'gpt-4'
        return response
    else:
        response = llm.generate(messages=messages, max_tokens=200, streaming=streaming)
        def stream():
            for token in response:
                completion_timestamp = int(time.time())
                completion_id = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=28))

                completion_data = {
                    'id': f'chatcmpl-{completion_id}',
                    'object': 'chat.completion.chunk',
                    'created': completion_timestamp,
                    'model': model,
                    'choices': [
                        {
                            'delta': {
                                'content': token
                            },
                            'index': 0,
                            'finish_reason': None
                        }
                    ]
                }

                yield 'data: %s\n\n' % json.dumps(completion_data, separators=(',' ':'))
                time.sleep(0.1)

        return app.response_class(stream(), mimetype='text/event-stream')


def output(chunk):
    if b'"youChatToken"' in chunk:
        chunk_json = json.loads(chunk.decode().split('data: ')[1])
        print(chunk_json['youChatToken'], flush=True, end = '')

if __name__ == '__main__':
    config = {
        'host': '0.0.0.0',
        'port': 1337,
        'debug': True
    }

    app.run(**config)




