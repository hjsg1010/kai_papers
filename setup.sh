#!/bin/bash

# AI Paper Newsletter - Setup Script
# This script helps you set up the entire system

set -e

echo "🚀 AI Paper Newsletter - Setup Script"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created. Please edit it with your credentials.${NC}"
    echo ""
    echo "Required credentials:"
    echo "  - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE_KEY"
    echo ""
    read -p "Press Enter after you've updated .env file..."
fi

# Check Python version
echo "📍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✅ Python $python_version found${NC}"
else
    echo -e "${RED}❌ Python 3.11+ is required. Current: $python_version${NC}"
    exit 1
fi

# Check Docker
echo "📍 Checking Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker found${NC}"
    docker_available=true
else
    echo -e "${YELLOW}⚠️  Docker not found. Will use local installation.${NC}"
    docker_available=false
fi

# Check Docker Compose
if [ "$docker_available" = true ]; then
    echo "📍 Checking Docker Compose..."
    if command -v docker-compose &> /dev/null; then
        echo -e "${GREEN}✅ Docker Compose found${NC}"
        docker_compose_available=true
    else
        echo -e "${YELLOW}⚠️  Docker Compose not found${NC}"
        docker_compose_available=false
    fi
fi

# Installation method selection
echo ""
echo "Choose installation method:"
echo "1. Docker Compose (recommended)"
echo "2. Local installation"
read -p "Enter choice (1 or 2): " install_choice

if [ "$install_choice" = "1" ]; then
    if [ "$docker_compose_available" = true ]; then
        echo ""
        echo "🐳 Installing with Docker Compose..."
        
        # Build and start services
        docker-compose build
        docker-compose up -d
        
        echo -e "${GREEN}✅ Services started successfully!${NC}"
        echo ""
        echo "Services:"
        echo "  - Paper Processor: http://localhost:8000"
        echo "  - n8n: http://localhost:5678"
        echo ""
        echo "Check status: docker-compose ps"
        echo "View logs: docker-compose logs -f"
        
    else
        echo -e "${RED}❌ Docker Compose is required for this option${NC}"
        exit 1
    fi
    
elif [ "$install_choice" = "2" ]; then
    echo ""
    echo "🔧 Local installation..."
    
    # Create virtual environment
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
    
    # Install system dependencies
    echo "📦 Checking system dependencies..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing poppler (for PDF processing)..."
            brew install poppler
        else
            echo -e "${YELLOW}⚠️  Homebrew not found. Please install poppler manually:${NC}"
            echo "    brew install poppler"
        fi
    else
        # Linux
        echo "Installing poppler-utils (for PDF processing)..."
        sudo apt-get update
        sudo apt-get install -y poppler-utils
    fi
    
    # Check if n8n should be installed
    echo ""
    read -p "Do you want to install n8n? (y/n): " install_n8n
    
    if [ "$install_n8n" = "y" ]; then
        echo "📦 Installing n8n..."
        npm install -g n8n
        echo -e "${GREEN}✅ n8n installed${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}✅ Local installation complete!${NC}"
    echo ""
    echo "To start the services:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Start Paper Processor: python paper_processor.py"
    echo "  3. Start n8n (in another terminal): n8n start"
    
else
    echo -e "${RED}❌ Invalid choice${NC}"
    exit 1
fi

# Test the installation
echo ""
read -p "Do you want to test the installation? (y/n): " test_install

if [ "$test_install" = "y" ]; then
    echo "🧪 Testing Paper Processor..."
    sleep 3
    
    # Test health endpoint
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ Paper Processor is running correctly!${NC}"
    else
        echo -e "${YELLOW}⚠️  Paper Processor might not be ready yet. HTTP Status: $response${NC}"
        echo "Try accessing http://localhost:8000/docs for API documentation"
    fi
fi

# Create S3 bucket (optional)
echo ""
read -p "Do you want to create the S3 bucket now? (y/n): " create_bucket

if [ "$create_bucket" = "y" ]; then
    source .env
    echo "Creating S3 bucket: $S3_BUCKET_NAME"
    aws s3 mb s3://$S3_BUCKET_NAME || echo "Bucket might already exist"
    aws s3api put-object --bucket $S3_BUCKET_NAME --key ${S3_PAPERS_PREFIX} || true
    echo -e "${GREEN}✅ S3 bucket configured${NC}"
fi

echo ""
echo "======================================"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Import n8n workflow:"
echo "   - Open http://localhost:5678"
echo "   - Go to Workflows → Import from File"
echo "   - Select n8n_workflow.json"
echo ""
echo "2. Configure n8n credentials:"
echo "   - AWS S3"
echo "   - Slack (optional)"
echo ""
echo "3. Test the workflow manually or wait for scheduled run"
echo ""
echo "📚 For more information, see README.md"
echo ""