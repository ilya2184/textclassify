# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13-slim

EXPOSE 5050

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app
COPY ./check_ping.sh /usr/local/bin/check_ping.sh

RUN adduser -u 5678 --disabled-password --gecos "" appuser \
    && chown -R appuser /app \
    && chmod +x /usr/local/bin/check_ping.sh
USER appuser
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

VOLUME ["/app/models"]

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "main:app"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/usr/local/bin/check_ping.sh"]
