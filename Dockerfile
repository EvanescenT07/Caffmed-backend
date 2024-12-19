# Use a more complete base image with CUDA support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "model.py"]