docker build -t ladder-ai-image .  
docker run -it -p 5555:5555 --rm --name ladder-ai ladder-ai-image
