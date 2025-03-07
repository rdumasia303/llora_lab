# Use Nginx as base image
FROM nginx:alpine

# Copy the static HTML/CSS/JS files
COPY index.html /usr/share/nginx/html/
COPY dist/ /usr/share/nginx/html/

# Copy NGINX configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start NGINX
CMD ["nginx", "-g", "daemon off;"]

