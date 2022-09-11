docker build -t ladder-ai-image .  
docker run -it -p 5000:5000 --rm --name ladder-ai ladder-ai-image
