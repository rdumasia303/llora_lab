# Llora Lab

<div align="center">
  <img src="docs/logo.png" alt="Llora Lab Logo" width="200"/>
  
  <p>A comprehensive platform for fine-tuning, managing, and deploying large language models with VLLM, openweb-ui and Unsloth.</p>

  <p>
    <a href="#key-features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#troubleshooting">Troubleshooting</a>
  </p>
</div>

---

## Overview

Llora Lab provides a complete environment for experimenting with large language models - from dataset preparation to fine-tuning with LoRA adapters to deployment and testing. It combines user-friendly interfaces with powerful backend capabilities, making advanced LLM workflows accessible to both researchers and developers.

### Why Llora Lab?

- **Simplified Workflows**: From raw datasets to deployed models in just a few clicks
- **Resource Efficiency**: Fine-tune powerful models with affordable hardware requirements
- **Complete Solution**: Everything you need for the full LLM lifecycle in one integrated platform
- **Docker-based**: Easy deployment with containers that handle the complexity for you

## Key Features

- **Model Management**: Import, configure, and organize LLM models from Hugging Face
- **LoRA Adapter Training**: Create and train efficient adapters on custom datasets
- **Dataset Handling**: Upload, preview, and manage training datasets in JSONL format
- **Serving Interface**: Deploy models with an OpenAI-compatible API endpoint
- **Testing Environment**: Test models directly within the UI or via API
- **System Monitoring**: Track GPU usage, memory, and container status
- **Real-time Logs**: Access training and serving logs in real-time

## Architecture

Llora Lab is built as a containerized application with several core components:

- **Admin API**: FastAPI backend that orchestrates the entire system
- **Admin UI**: React-based interface for managing all operations
- **Trainer**: Container for running model fine-tuning jobs
- **vLLM Server**: High-performance inference server for model deployment
- **Open WebUI**: Chat interface for interacting with deployed models

![Llora Lab Architecture](docs/architecture.png)

## Installation

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed (for GPU access)
- 16GB+ system RAM (32GB+ recommended)
- 100GB+ disk space for models and datasets

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llora-lab.git
   cd llora-lab
   ```

2. Create a `.env` file with your Hugging Face token:
   ```bash
   echo "HF_TOKEN=your_huggingface_token" > .env
   ```

3. Build and start the services:
   ```bash
   make build
   make build-ui  # Build the admin UI frontend
   make start
   ```

4. Access the admin interface at http://localhost:3001

### Docker Compose Manual Setup

Alternatively, you can use Docker Compose directly:

```bash
# Build the images
docker compose build admin-api trainer vllm

# Start the admin services
docker compose up -d admin-api admin-ui
```

## Usage

### Workflow Overview

1. **Configure Models**: Add model configurations from Hugging Face
2. **Upload Datasets**: Prepare and upload training data in JSONL format
3. **Create Adapters**: Configure LoRA adapters for your models
4. **Train Adapters**: Start training jobs with your datasets
5. **Deploy Models**: Serve models with or without adapters
6. **Test Models**: Interact with your deployed models through the UI or API

### Adding Models

1. Navigate to the "Models" tab
2. Click "Add Model"
3. Enter the Hugging Face model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
4. Configure model parameters as needed
5. Click "Save Model"

### Uploading Datasets

1. Navigate to the "Datasets" tab
2. Click "Upload Dataset"
3. Select a JSONL file with your training data
4. Wait for validation and processing
5. Preview the dataset to ensure proper formatting

### Creating and Training Adapters

1. Navigate to the "Adapters" tab
2. Click "Create Adapter"
3. Select a base model and dataset
4. Configure LoRA parameters (rank, alpha, etc.)
5. Click "Start Training"
6. Monitor progress in the "Training" tab

### Deploying Models

1. Navigate to the "Serving" tab
2. Select a model and optionally an adapter
3. Click "Start Serving"
4. Wait for initialization to complete
5. Access your model via the API or the integrated chat UI

### Testing Models

- Use the built-in testing interface in the "Serving" tab
- Access the OpenWebUI chat interface at http://localhost:3000
- Connect via the OpenAI-compatible API at http://localhost:8000/v1

## Configuration

### Environment Variables

Create a `.env` file with these options:

```bash
# Required
HF_TOKEN=your_huggingface_token

# Optional
LOG_LEVEL=info             # Log level (debug, info, warning, error)
CORS_ORIGINS=*             # CORS allowed origins
CUDA_VERSION=124           # CUDA version for the trainer
```

### Model Configuration

Model configs support these parameters:

- **name**: Unique identifier for the model
- **model_id**: HuggingFace model ID
- **quantization**: Quantization method (bitsandbytes, awq, gptq, gguf)
- **max_model_len**: Maximum sequence length
- **gpu_memory_utilization**: GPU memory usage (0.0-1.0)
- **tensor_parallel_size**: Number of GPUs for tensor parallelism

### Adapter Configuration

Adapter configs support these parameters:

- **name**: Unique identifier for the adapter
- **base_model**: Reference to a configured model
- **dataset**: Training dataset filename
- **lora_rank**: LoRA rank parameter (typically 8-64)
- **lora_alpha**: LoRA alpha parameter (typically 16-32)
- **steps**: Number of training steps
- **learning_rate**: Learning rate for training

## Troubleshooting

### GPU Issues

- Ensure NVIDIA drivers are installed and up-to-date
- Verify the NVIDIA Container Toolkit is properly installed
- Run `nvidia-smi` to confirm GPU is accessible
- Check GPU memory availability before starting jobs

### Container Issues

- View container logs: `make logs service=admin-api`
- Check container status: `docker compose ps`
- If containers are stuck, try stopping and restarting: `make stop && make start`

### Training Problems

- Verify dataset format is correct JSONL
- Check adapter parameters are appropriate for your hardware
- Ensure sufficient disk space for model weights
- Review logs in the UI or with `make logs service=trainer`

### Serving Issues

- Make sure there's enough GPU memory for the selected model
- Verify no other serving containers are running
- Check network connectivity between containers
- Review logs in the UI or with `make logs service=vllm`

## Development

### Project Structure

```
llora-lab/
├── admin/              # Admin API backend
├── admin-ui/           # React frontend
├── configs/            # Model and adapter configurations
├── docker/             # Dockerfiles for services
├── datasets/           # Training datasets
├── adapters/           # Trained adapters
├── logs/               # Log files
├── huggingface-cache/  # Cached model files
└── scripts/            # Utility scripts
```

### Building the UI

The UI uses Vite and React:

```bash
cd admin-ui
npm install
npm run dev     # Development mode
npm run build   # Production build
```

### Admin API Development

The Admin API uses FastAPI:

```bash
cd admin
pip install -r requirements.txt
uvicorn main:app --reload  # Development mode
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- [Open WebUI](https://github.com/open-webui/open-webui) for the chat interface
- [Hugging Face](https://huggingface.co/) for model hosting and libraries
- [FastAPI](https://fastapi.tiangolo.com/) and [React](https://reactjs.org/) for the tech stack
