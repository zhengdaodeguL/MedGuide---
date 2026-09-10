FROM caddy:2.10-alpine

COPY deploy/gateway/Caddyfile /etc/caddy/Caddyfile
COPY deploy/gateway/entrypoint.sh /usr/local/bin/medguide-gateway
RUN chmod +x /usr/local/bin/medguide-gateway

EXPOSE 80 443
ENTRYPOINT ["/usr/local/bin/medguide-gateway"]
