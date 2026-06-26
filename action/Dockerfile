FROM python:3.11-slim

ARG PRETIA_VERSION=pretia
RUN pip install --no-cache-dir ${PRETIA_VERSION}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
