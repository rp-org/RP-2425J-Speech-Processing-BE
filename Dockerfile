# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Define build arguments for environment variables
# ARG PROJECT_ID
# ARG SUBSCRIPTION_NAME

# Set environment variables during the build process
# ENV PROJECT_ID=$PROJECT_ID
# ENV SUBSCRIPTION_NAME=$SUBSCRIPTION_NAME


# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements.txt to the container
COPY requirements.txt .

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Set the environment variable for FastAPI
ENV PYTHONUNBUFFERED=1

# Expose Port
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]