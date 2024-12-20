# Use TensorFlow's official base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY .env . 
COPY requirements.txt .
COPY brain_tumorV2.h5 .
COPY model.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Switch to a non-root user for security for debian
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 5000

# Command to run the app
CMD ["python", "model.py"]
