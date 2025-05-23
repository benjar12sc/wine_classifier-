# Dockerfile for training and serving the SVM model
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command just prints help
CMD ["python", "scripts/train_svm.py", "--help"]
