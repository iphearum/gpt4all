# docker build -t api_ai_chatbot:tag -f Dockerfile . #1337
GPT="iphearum/gpt4all:tag"
# docker push ${GPT}
docker image build -t ${GPT} 1337:1337 -f Dockerfile . #5000

# docker run -d -p 5000 api_ai_chatbot:tag
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

# PORT=5000
# docker run -d -p ${PORT}:5000 ${GPT}:tag

COUNT=3
PORT=1337
for i in $(seq 1 $COUNT)
do
    docker run -d -p ${PORT}:1337 ${GPT}
    PORT=$((PORT+1))
done
# docker ps -al