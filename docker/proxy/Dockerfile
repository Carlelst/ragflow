FROM python:3.11-slim
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY proxy_server.py /app/
ENV RAGFLOW_BASE_URL=http://127.0.0.1:8088
ENV RAGFLOW_API_KEY=ragflow-307044760fae4f548209426ba6191d9e
WORKDIR /app
EXPOSE 8090
ENTRYPOINT ["gunicorn", "proxy_server:app", "--worker-class", "gevent", "--worker-connections", "1000", "--timeout", "120", "--max-requests", "50000", "--max-requests-jitter", "5000", "--keep-alive", "30", "--bind", "0.0.0.0:8090"]
CMD ["--workers", "4"]
