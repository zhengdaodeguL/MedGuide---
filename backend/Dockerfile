FROM python:3.13-slim

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY data/knowledge /app/data/knowledge
COPY scripts/build_bm25_index.py /app/scripts/build_bm25_index.py
COPY backend/docker-entrypoint.sh /usr/local/bin/medguide-entrypoint
RUN useradd --create-home --uid 10001 medguide \
  && mkdir -p /app/data/runtime \
  && touch /app/data/runtime/.volume-owner \
  && chmod +x /usr/local/bin/medguide-entrypoint \
  && chown -R medguide:medguide /app
ENV PYTHONPATH=/app/backend
USER medguide
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=2)" || exit 1
ENTRYPOINT ["/usr/local/bin/medguide-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
