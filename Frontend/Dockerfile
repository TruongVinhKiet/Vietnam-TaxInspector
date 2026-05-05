# ════════════════════════════════════════════════════════════════
#  Dockerfile – TaxInspector Frontend (Nginx Static Server)
# ════════════════════════════════════════════════════════════════
#  Container: tax-frontend-ui (Port 80 → mapped to 3000)
#  Role: Serve static HTML/CSS/JS + reverse proxy to API
# ════════════════════════════════════════════════════════════════

FROM nginx:alpine

LABEL maintainer="TaxInspector Team"
LABEL description="Frontend UI with Nginx reverse proxy"

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/taxinspector.conf

# Copy frontend files
COPY . /usr/share/nginx/html/

# Remove files that shouldn't be served
RUN rm -f /usr/share/nginx/html/Dockerfile \
          /usr/share/nginx/html/nginx.conf

EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
