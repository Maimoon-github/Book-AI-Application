#!/bin/bash

# RAG Book AI Docker Deployment Script

echo "🚀 Starting RAG Book AI Deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your actual API keys and settings!"
    read -p "Press Enter to continue after editing .env file..."
fi

# Build and start containers
echo "🏗️  Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Run Django migrations
echo "🔄 Running Django migrations..."
docker-compose exec web python manage.py migrate

# Create superuser (optional)
echo "👤 Do you want to create a Django superuser? (y/n)"
read -r create_superuser
if [ "$create_superuser" = "y" ]; then
    docker-compose exec web python manage.py createsuperuser
fi

# Show status
echo "✅ Deployment complete!"
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🌐 Access your application:"
echo "   - Main App: http://localhost"
echo "   - Django Admin: http://localhost/admin"
echo "   - Direct Django: http://localhost:8000"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart: docker-compose restart"
echo "   - Update: docker-compose pull && docker-compose up -d --build"
