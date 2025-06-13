# 🐳 RAG Book AI - Quick Docker Setup

## 🚀 One-Command Deployment

### Windows (PowerShell)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\deploy.ps1
```

### Alternative: Manual Steps
```powershell
# 1. Create environment file
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 2. Build and run (simple version)
docker-compose -f docker-compose.simple.yml up --build -d

# 3. Run migrations
docker-compose -f docker-compose.simple.yml exec web python manage.py migrate

# 4. Create superuser (optional)
docker-compose -f docker-compose.simple.yml exec web python manage.py createsuperuser
```

## 🌐 Access Your App
- **Application**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin

## 🔧 Management
```powershell
# View logs
docker-compose -f docker-compose.simple.yml logs -f

# Stop
docker-compose -f docker-compose.simple.yml down

# Restart
docker-compose -f docker-compose.simple.yml restart
```

## ⚙️ Configuration

### Required Environment Variables (.env):
```env
GROQ_API_KEY=your_groq_api_key_here
DEBUG=True
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=localhost,127.0.0.1
```

## 🚨 Troubleshooting

### If build fails:
```powershell
# Clean Docker cache
docker system prune -f
docker-compose -f docker-compose.simple.yml build --no-cache
```

### If container won't start:
```powershell
# Check logs
docker-compose -f docker-compose.simple.yml logs web

# Check container status
docker-compose -f docker-compose.simple.yml ps
```

## 📋 Files Created:
- `Dockerfile` - Container definition
- `docker-compose.yml` - Full production setup (with Nginx)
- `docker-compose.simple.yml` - Simple development setup
- `.env.example` - Environment variables template
- `deploy.ps1` - Automated deployment script
- `.dockerignore` - Files to exclude from build
