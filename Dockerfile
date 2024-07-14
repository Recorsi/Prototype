# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app .

# Install system packages needed
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-dev \
    git \
    wget \
    && apt-get clean

# Install dlib dependencies
RUN pip install --upgrade pip
RUN pip install numpy

# Install dlib from source
RUN git clone https://github.com/davisking/dlib.git \
    && cd dlib \
    && python setup.py install \
    && cd .. \
    && rm -rf dlib

# Install other Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the manually downloaded weights into the Docker image
COPY app/xception-b5690688.pth /root/.cache/torch/hub/checkpoints/

# Ensure the entrypoint script is executable
RUN chmod +x detect_from_video.py

# Expose port 5000
EXPOSE 5000

# Define the command to run the application
CMD ["python3", "main.py"]