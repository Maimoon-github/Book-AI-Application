# 🐳 RAG Book AI - Docker Deployment Guide

## 📋 Prerequisites

- Docker Desktop 4.37+ (with CLI support)
- At least 4GB RAM available
- GROQ API Key

## 🚀 Quick Start

### 1. Clone and Navigate
```bash
cd rag_book_ai
```

### 2. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 3. Deploy with One Command

**Windows:**
```cmd
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### 4. Access Your App
- **Main Application**: http://localhost
- **Django Admin**: http://localhost/admin
- **Direct Django**: http://localhost:8000

## 🏗️ Manual Deployment

### Build and Start
```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser (optional)
docker-compose exec web python manage.py createsuperuser
```

## 📊 Container Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Nginx    │    │   Django    │    │    Redis    │
│   (Port 80) │───▶│  (Port 8000)│───▶│ (Port 6379) │
│   Reverse   │    │   Web App   │    │   Caching   │
│    Proxy    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🔧 Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f web
docker-compose logs -f nginx
docker-compose logs -f redis
```

### Stop/Start Services
```bash
# Stop all
docker-compose down

# Start all
docker-compose up -d

# Restart specific service
docker-compose restart web
```

### Database Operations
```bash
# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Access Django shell
docker-compose exec web python manage.py shell

# Backup database
docker-compose exec web python manage.py dumpdata > backup.json
```

### Update Application
```bash
# Pull latest images and rebuild
docker-compose pull
docker-compose up -d --build

# Or rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🔐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your GROQ API key | Required |
| `DEBUG` | Django debug mode | `False` |
| `SECRET_KEY` | Django secret key | Generated |
| `ALLOWED_HOSTS` | Allowed domains | `localhost,127.0.0.1` |

## 📁 Volume Mounts

| Container Path | Host Path | Purpose |
|---------------|-----------|---------|
| `/app/media` | `./media` | User uploads |
| `/app/db.sqlite3` | `./db.sqlite3` | Database |
| `/data` | `redis_data` | Redis persistence |

## 🚨 Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs web

# Rebuild container
docker-compose build --no-cache web
docker-compose up -d
```

### Database Issues
```bash
# Reset database
docker-compose down
rm db.sqlite3
docker-compose up -d
docker-compose exec web python manage.py migrate
```

### Port Conflicts
```bash
# Check what's using port 80
netstat -ano | findstr :80    # Windows
lsof -i :80                   # Linux/Mac

# Use different ports in docker-compose.yml
ports:
  - "8080:80"  # Change from 80 to 8080
```

### Permission Issues (Linux/Mac)
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x deploy.sh
```

## 🔄 Production Deployment

### 1. Update Settings
```python
# In settings.py
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'www.your-domain.com']
```

### 2. Use Production Database
```yaml
# Add PostgreSQL to docker-compose.yml
db:
  image: postgres:15
  environment:
    POSTGRES_DB: bookaidb
    POSTGRES_USER: bookaiuser
    POSTGRES_PASSWORD: your_password
```

### 3. SSL/HTTPS Setup
```yaml
# Add SSL certificates to nginx
volumes:
  - ./ssl/cert.pem:/etc/ssl/certs/cert.pem
  - ./ssl/key.pem:/etc/ssl/private/key.pem
```

## 📈 Monitoring

### Health Checks
```bash
# Check container health
docker-compose ps

# Test application endpoint
curl http://localhost/health/
```

### Resource Usage
```bash
# Monitor resource usage
docker stats

# Check disk usage
docker system df
```

## 🧹 Cleanup

### Remove Everything
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi $(docker images -q)

# Remove volumes (⚠️ This deletes data!)
docker-compose down -v
```

### Prune System
```bash
# Clean up unused resources
docker system prune -af
```

## 📞 Support

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Verify environment variables in `.env`
3. Ensure Docker Desktop is running
4. Check port availability
5. Restart containers: `docker-compose restart`
