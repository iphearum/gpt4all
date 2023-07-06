import gpt4all

models = [
    # finished
    # {"filename": "ggml-mpt-7b-instruct.bin"},
    # {"filename": "ggml-v3-13b-hermes-q5_1.bin"},
    {"filename": "ggml-gpt4all-j-v1.3-groovy.bin"},

    # needed for download
    # {"filename": "ggml-gpt4all-l13b-snoozy.bin"},
    # {"filename": "ggml-mpt-7b-chat.bin"},
]

class GPT:
    
    def __init__(self):
        pass

    def main(self):
        for i in models:
            return gpt4all.GPT4All(i['filename'])

if __name__ == '__main__':
#    GPT().main()
    gpt = GPT()
