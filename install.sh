#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}     Llora Lab Installation Script    ${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# Check requirements
echo -e "${BLUE}Checking requirements...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is required but not installed.${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose is required but not installed.${NC}"
    echo "Please install Docker Compose first: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is installed${NC}"

# Check NVIDIA GPU and drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠ NVIDIA tools not found. GPU acceleration may not be available.${NC}"
    echo -e "${YELLOW}⚠ For best performance, install NVIDIA drivers and NVIDIA Container Toolkit.${NC}"
    read -p "Continue without GPU support? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    # Check NVIDIA Container Toolkit
    if ! docker info | grep -q "Runtimes.*nvidia"; then
        echo -e "${YELLOW}⚠ NVIDIA Container Toolkit not properly set up.${NC}"
        echo "Please install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        read -p "Continue without GPU support? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ NVIDIA Container Toolkit is installed${NC}"
    fi
fi

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p admin admin-ui docker/requirements scripts configs datasets adapters logs huggingface-cache

# Create requirements files
echo -e "${BLUE}Creating requirements files...${NC}"

# Common requirements
cat > docker/requirements/common.txt << EOF
torch>=2.4.0
transformers>=4.39.0
accelerate>=0.31.0
huggingface-hub>=0.20.0
safetensors>=0.4.2
datasets>=2.17.0
numpy>=1.26.0
pydantic>=2.5.0
tqdm>=4.66.0
sentencepiece>=0.1.99
einops>=0.7.0
EOF

# Training requirements
cat > docker/requirements/train.txt << EOF
bitsandbytes>=0.43.0
trl>=0.8.0
peft>=0.9.0
wandb>=0.16.0
evaluate>=0.4.0
tensorboard>=2.15.0
psutil>=5.9.0
matplotlib>=3.7.0
pandas>=2.0.0
EOF

# Serving requirements
cat > docker/requirements/serve.txt << EOF
vllm>=0.3.2
xformers>=0.0.23
optimum>=1.17.0
fastapi>=0.105.0
uvicorn>=0.24.0.post1
ray>=2.9.0
openai>=1.9.0
prometheus-client>=0.18.0
EOF

# Admin requirements
cat > admin/requirements.txt << EOF
fastapi>=0.105.0
uvicorn>=0.24.0.post1
docker>=7.0.0
pydantic>=2.5.0
toml>=0.10.2
python-multipart>=0.0.7
httpx>=0.26.0
psutil>=5.9.0
EOF

# Create sample config files
echo -e "${BLUE}Creating sample config files...${NC}"

# Sample model config
cat > configs/llama-3.1-8b.toml << EOF
# Model Configuration for Llama 3.1 8B
model_id = "meta-llama/Llama-3.1-8B-Instruct"
description = "Meta Llama 3.1 8B Instruct"

# Quantization settings
quantization = "bitsandbytes"
load_format = "bitsandbytes"

# Model dimensions
max_model_len = 8192
max_num_seqs = 16

# Performance settings
gpu_memory_utilization = 0.99
enforce_eager = false

# Additional parameters
[additional_params]
chat_template = "llama-3.1"
response_role = "assistant"
EOF

# Create the admin API app
echo -e "${BLUE}Creating admin API app...${NC}"
cp -f admin/app.py admin/

# Create the admin-ui Nginx config
echo -e "${BLUE}Creating admin UI Nginx config...${NC}"
cp -f admin-ui/nginx.conf admin-ui/

# Copy Dockerfiles
echo -e "${BLUE}Copying Dockerfiles...${NC}"
cp -f admin/Dockerfile admin/
cp -f admin-ui/Dockerfile admin-ui/
cp -f docker/Dockerfile.trainer docker/
cp -f docker/Dockerfile.vllm docker/

# Copy training scripts
echo -e "${BLUE}Copying training scripts...${NC}"
mkdir -p scripts/utils
cp -f scripts/train.py scripts/
cp -f scripts/test_adapter.py scripts/

# Create .env file
echo -e "${BLUE}Creating .env file...${NC}"
cat > .env << EOF
# Environment Variables for Llora Lab
LOG_LEVEL=info
CORS_ORIGINS=*
CUDA_VERSION=124
EOF

# Copy docker-compose.yml
echo -e "${BLUE}Copying docker-compose.yml and Makefile...${NC}"
cp -f docker-compose.yml ./
cp -f Makefile ./

echo -e "${GREEN}Installation complete!${NC}"
echo -e "To start Llora Lab, run: ${YELLOW}make start${NC}"
echo -e "Then access the admin UI at: ${YELLOW}http://localhost:3001${NC}"
echo ""
echo -e "${BLUE}Happy fine-tuning!${NC}"
