# Use TensorFlow's official base image
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      PORT=5000

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY brain_tumorV2.h5 .
COPY model.py .


# Switch to a non-root user for security for debian
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5000", "model:app", "--timeout", "120"]
