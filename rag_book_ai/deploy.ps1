# RAG Book AI Docker Deployment PowerShell Script

Write-Host "Starting RAG Book AI Deployment..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Please edit .env file with your actual API keys and settings!" -ForegroundColor Yellow
    Read-Host "Press Enter to continue after editing .env file"
}

# Build and start containers
Write-Host "Building Docker containers..." -ForegroundColor Blue
docker-compose build

Write-Host "Starting services..." -ForegroundColor Blue
docker-compose up -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep 15

# Run Django migrations
Write-Host "Running Django migrations..." -ForegroundColor Blue
docker-compose exec web python manage.py migrate

# Create superuser (optional)
$createSuperuser = Read-Host "Do you want to create a Django superuser? (y/n)"
if ($createSuperuser -eq "y") {
    docker-compose exec web python manage.py createsuperuser
}

# Show status
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "Access your application:" -ForegroundColor Cyan
Write-Host "   - Main App: http://localhost" -ForegroundColor White
Write-Host "   - Django Admin: http://localhost/admin" -ForegroundColor White
Write-Host "   - Direct Django: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "   - View logs: docker-compose logs -f" -ForegroundColor White
Write-Host "   - Stop services: docker-compose down" -ForegroundColor White
Write-Host "   - Restart: docker-compose restart" -ForegroundColor White
Write-Host "   - Update: docker-compose pull; docker-compose up -d --build" -ForegroundColor White

Read-Host "Press Enter to exit"
