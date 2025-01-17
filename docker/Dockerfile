# Use official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy files to container
COPY requirements.txt ./
COPY scripts/ ./scripts

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Set entrypoint
CMD ["python", "scripts/train_model.py"]