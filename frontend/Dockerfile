FROM node:24-alpine AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/templates/default.conf.template
COPY frontend/docker-entrypoint.d/05-medguide-validate.sh /docker-entrypoint.d/05-medguide-validate.sh
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
RUN chmod +x /docker-entrypoint.d/05-medguide-validate.sh
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1/_health \
  && wget -q -O /dev/null http://127.0.0.1/_api-ready || exit 1
