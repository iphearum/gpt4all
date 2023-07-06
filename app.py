
import gpt4all
from flask import Flask, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

models = [
    # finished
    # {"filename": "ggml-mpt-7b-instruct.bin"},
    # {"filename": "ggml-v3-13b-hermes-q5_1.bin"},
    {"filename": "ggml-gpt4all-j-v1.3-groovy.bin"},

    # needed for download
    # {"filename": "ggml-gpt4all-l13b-snoozy.bin"},
    # {"filename": "ggml-mpt-7b-chat.bin"},
]
MODELPATH = "../models/gpt4all/"
for i in models:
    gpt = gpt4all.GPT4All(i['filename'],MODELPATH)
# gpt


class ChatGPT(Resource):
    sysprompt = f"""
    ### Instruction: 
    You are OpenBrain, a large language model. Follow the user's instructions carefully. Markdown will be used to respond.
    ### Prompt: 
    """

    def __init__(self):
        self.messages = [{"role": "system", "content": f"{self.sysprompt}"}]



    def get(self):
        return self.messages

    def post(self):
        args = request.json
        for msg in args:
            self.messages.append(msg)
        response = gpt.chat_completion(self.messages, default_prompt_header=False, streaming=True)
        # return Response(response)
        response["model"] = 'gpt-4'
        # return response
        # name = args['name']
        # age = args['age']
        # return {'name': name, 'age': age}
        # return args
        # for t in args:
        #     messages.append(t)

        # response = gpt.chat_completion(messages, default_prompt_header=False)
        # response["model"] = 'gpt-4'
        return response

    def put(self):
        pass

    def delete(self):
        pass

api.add_resource(ChatGPT,"/chat/completions")

if __name__ == '__main__':
    app.run(debug=True)