https://medium.com/@jonas.granlund/docker-essentials-building-and-running-your-first-container-47aff380b50b

Run container and share the current folder: docker run --name mediapipe -d -v .:/app -i  python

- --name: name of container
- -v: the container's 'app' folder and the current folder are connected.
- -i: keep it running
- python: the image

Run a shell in the container: docker exec -it mediapipe bash

apt-get update && apt-get install -y python3-opencv

pip install mediapipe opencv-python
