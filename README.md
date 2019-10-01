# Neural Rules

This repository is for code related to the neural network rule project.

Requires Tensorflow, Pandas, Jupyter, and Python 2.7.


## Notes on Environment Setup

Using Tensorflow with a GPU requires extra care during Docker installation. Specifically, make sure the NVIDIA drivers are at least 10.1 and that the NVIDIA Docker runtime is correctly installed.

For creating (and then restarting) a Docker image, use:

    sudo docker run --runtime=nvidia -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter /bin/bash
    sudo docker ps -a
    sudo docker start -i <image-name>

Within the Docker image, the following command is used to initiate the Jupyter server:

    bash -c 'source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root'
