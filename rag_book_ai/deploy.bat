@echo off
REM RAG Book AI Docker Deployment Script for Windows

echo 🚀 Starting RAG Book AI Deployment...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your actual API keys and settings!
    pause
)

REM Build and start containers
echo 🏗️  Building Docker containers...
docker-compose build

echo 🚀 Starting services...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10

REM Run Django migrations
echo 🔄 Running Django migrations...
docker-compose exec web python manage.py migrate

REM Create superuser (optional)
set /p create_superuser="👤 Do you want to create a Django superuser? (y/n): "
if /i "%create_superuser%"=="y" (
    docker-compose exec web python manage.py createsuperuser
)

REM Show status
echo ✅ Deployment complete!
echo.
echo 📊 Service Status:
docker-compose ps

echo.
echo 🌐 Access your application:
echo    - Main App: http://localhost
echo    - Django Admin: http://localhost/admin
echo    - Direct Django: http://localhost:8000
echo.
echo 📋 Useful commands:
echo    - View logs: docker-compose logs -f
echo    - Stop services: docker-compose down
echo    - Restart: docker-compose restart
echo    - Update: docker-compose pull ^&^& docker-compose up -d --build

pause
