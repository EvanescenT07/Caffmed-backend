FROM python:3.9-alpine

RUN apk update && apk add --no-cache shadow

RUN addgroup -S appgroup && \
      adduser -S appuser -G appgroup


USER appuser

WORKDIR /app

COPY .env .
COPY requirements.txt .
COPY brain_tumorV2.h5 .
COPY model.py .

EXPOSE 5000

CMD ["python", "model.py"]

