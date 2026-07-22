FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./admin_ui
EXPOSE 8000
CMD ["uvicorn", "admin_ui.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
