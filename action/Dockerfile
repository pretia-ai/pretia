FROM python:3.11-slim

ARG AGENTCOST_VERSION=agentcost
RUN pip install --no-cache-dir ${AGENTCOST_VERSION}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
