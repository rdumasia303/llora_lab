from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi.background import BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union, Literal
import docker
import toml
import os
import shutil
import logging
import json
import uuid
import asyncio
import httpx
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llora-lab")


# Shared variable to store latest GPU stats
latest_gpu_stats = {
    "gpus": [],
    "utilized": "0.0",
    "temperature": "N/A",
    "memory": "0/0 MB",
    "last_updated": 0
}

# Initialize FastAPI app
app = FastAPI(title="Llora Lab API", description="API for managing LLM training and serving")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Docker client
docker_client = docker.from_env()

# Get Hugging Face token from environment
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    logger.warning("No Hugging Face token found in environment. Model downloading and training may fail.")

# Define base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
ADAPTER_DIR = os.path.join(BASE_DIR, "adapters")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HF_CACHE_DIR = os.path.join(BASE_DIR, "huggingface-cache")

# Ensure directories exist
for dir_path in [CONFIG_DIR, DATASET_DIR, ADAPTER_DIR, LOG_DIR, HF_CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Shared docker network name
DOCKER_NETWORK_NAME = "llora-lab-network"

# Active training and serving jobs
active_jobs = {}

# Model Schemas
class ModelConfig(BaseModel):
    """Configuration for a model to be served with vLLM"""
    name: str
    description: Optional[str] = None
    model_id: str  # HuggingFace model ID
    
    # vLLM parameters - all optional to allow full flexibility
    quantization: Optional[str] = None
    load_format: Optional[str] = None
    dtype: Optional[str] = None
    
    # Model dimensions
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    
    # Performance settings
    gpu_memory_utilization: Optional[float] = None
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    
    # Multi-GPU settings
    gpu_memory_utilization: Optional[float] = None
    
    # Special flags
    enforce_eager: Optional[bool] = None
    trust_remote_code: Optional[bool] = None
    
    # Specialized configuration
    kv_cache_dtype: Optional[str] = None
    rope_scaling: Optional[str] = None
    rope_theta: Optional[float] = None
    
    # Chat template configuration
    chat_template: Optional[str] = None
    response_role: Optional[str] = None
    
    # Additional arbitrary parameters to handle any vLLM option
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class AdapterConfig(BaseModel):
    """Configuration for a LoRA adapter"""
    name: str
    description: Optional[str] = None
    base_model: str  # Reference to a model config name
    dataset: str  # Path to dataset
    
    # Training hyperparameters
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    
    steps: Optional[int] = None
    batch_size: Optional[int] = None
    gradient_accumulation: Optional[int] = None
    learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None
    
    # Advanced options
    max_seq_length: Optional[int] = None
    chat_template: Optional[str] = None
    use_nested_quant: Optional[bool] = None


class TrainingJob(BaseModel):
    """Information about a training job"""
    id: str
    adapter_config: str
    status: str  # initializing, running, completed, failed, stopped
    start_time: str
    step: Optional[int] = 0
    total_steps: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    message: Optional[str] = None


class ServingJob(BaseModel):
    """Information about a model serving job"""
    id: str
    model_conf: str
    model_id: str
    adapter: Optional[str] = None
    status: str  # initializing, running, ready, failed, stopped
    start_time: str
    requests_served: int = 0
    avg_response_time: float = 0.0
    message: Optional[str] = None


class SystemMetrics(BaseModel):
    """System metrics for monitoring"""
    gpu_utilization: List[float]
    memory_usage: Dict[str, Any]
    containers: List[Dict[str, Any]]
    disk_usage: Dict[str, Any]
    

# Helper functions
def ensure_docker_network():
    """Ensure the shared Docker network exists"""
    try:
        docker_client.networks.get(DOCKER_NETWORK_NAME)
    except docker.errors.NotFound:
        logger.info(f"Creating Docker network: {DOCKER_NETWORK_NAME}")
        docker_client.networks.create(DOCKER_NETWORK_NAME, driver="bridge")


def format_container_logs(logs_text):
    """Format container logs for better readability"""
    if not logs_text:
        return "No logs available"
    
    # Process logs with timestamps
    try:
        lines = logs_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # Extract timestamp if present
            if ' ' in line and line[0].isdigit():
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    timestamp, message = parts
                    # Format timestamp if valid
                    try:
                        dt = datetime.fromisoformat(timestamp.strip())
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        formatted_lines.append(f"[{formatted_time}] {message}")
                    except:
                        formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    except:
        # If any error occurs, return the original text
        return logs_text



@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())



# API Routes
@app.get("/")
async def root():
    return {
        "message": "Llora Lab API is running",
        "version": "1.0.0",
        "hf_token_configured": bool(HF_TOKEN)
    }


# Configuration Management
@app.get("/configs/models", response_model=List[ModelConfig])
async def list_model_configs():
    """List all available model configurations"""
    configs = []
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith(".toml") and not filename.startswith("adapter_"):
            config_path = os.path.join(CONFIG_DIR, filename)
            try:
                config = toml.load(config_path)
                config["name"] = filename.replace(".toml", "")
                configs.append(ModelConfig(**config))
            except Exception as e:
                logger.error(f"Error loading config {filename}: {str(e)}")
    return configs


@app.post("/configs/models", response_model=ModelConfig)
async def create_model_config(config: ModelConfig):
    """Create a new model configuration"""
    config_path = os.path.join(CONFIG_DIR, f"{config.name}.toml")
    if os.path.exists(config_path):
        raise HTTPException(status_code=400, detail=f"Config {config.name} already exists")
    
    config_dict = config.model_dump(exclude_none=True)  # Only include non-None values
    
    with open(config_path, "w") as f:
        toml.dump(config_dict, f)
    
    return config


@app.get("/configs/models/{name}", response_model=ModelConfig)
async def get_model_config(name: str):
    """Get a specific model configuration"""
    config_path = os.path.join(CONFIG_DIR, f"{name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config {name} not found")
    
    config = toml.load(config_path)
    config["name"] = name
    return ModelConfig(**config)


@app.put("/configs/models/{name}", response_model=ModelConfig)
async def update_model_config(name: str, config: ModelConfig):
    """Update a model configuration"""
    config_path = os.path.join(CONFIG_DIR, f"{name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config {name} not found")
    
    config_dict = config.model_dump(exclude_none=True)
    config_dict["name"] = name  # Ensure name consistency
    
    with open(config_path, "w") as f:
        toml.dump(config_dict, f)
    
    return config


@app.delete("/configs/models/{name}")
async def delete_model_config(name: str):
    """Delete a model configuration"""
    config_path = os.path.join(CONFIG_DIR, f"{name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Config {name} not found")
    
    os.remove(config_path)
    return {"message": f"Config {name} deleted"}


# Similar endpoints for adapter configurations
@app.get("/configs/adapters", response_model=List[AdapterConfig])
async def list_adapter_configs():
    """List all available adapter configurations"""
    configs = []
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith(".toml") and filename.startswith("adapter_"):
            config_path = os.path.join(CONFIG_DIR, filename)
            try:
                config = toml.load(config_path)
                config["name"] = filename.replace("adapter_", "").replace(".toml", "")
                configs.append(AdapterConfig(**config))
            except Exception as e:
                logger.error(f"Error loading adapter config {filename}: {str(e)}")
    return configs


@app.post("/configs/adapters", response_model=AdapterConfig)
async def create_adapter_config(config: AdapterConfig):
    """Create a new adapter configuration"""
    config_path = os.path.join(CONFIG_DIR, f"adapter_{config.name}.toml")
    if os.path.exists(config_path):
        raise HTTPException(status_code=400, detail=f"Adapter config {config.name} already exists")
    
    config_dict = config.model_dump(exclude_none=True)
    
    with open(config_path, "w") as f:
        toml.dump(config_dict, f)
    
    return config


@app.get("/configs/adapters/{name}", response_model=AdapterConfig)
async def get_adapter_config(name: str):
    """Get a specific adapter configuration"""
    config_path = os.path.join(CONFIG_DIR, f"adapter_{name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Adapter config {name} not found")
    
    config = toml.load(config_path)
    config["name"] = name
    return AdapterConfig(**config)


@app.delete("/configs/adapters/{name}")
async def delete_adapter_config(name: str):
    """Delete an adapter configuration"""
    config_path = os.path.join(CONFIG_DIR, f"adapter_{name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Adapter config {name} not found")
    
    os.remove(config_path)
    return {"message": f"Adapter config {name} deleted"}


# Dataset Management
@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a JSONL dataset file"""
    # Validate file extension
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only JSONL files are supported")
    
    # Save the file
    file_path = os.path.join(DATASET_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Validate JSONL format
    try:
        line_count = 0
        with open(file_path, "r") as f:
            for line in f:
                json.loads(line)
                line_count += 1
    except json.JSONDecodeError:
        # Try to fix the JSONL file
        try:
            fixed_path = os.path.join(DATASET_DIR, f"fixed_{file.filename}")
            fix_jsonl(file_path, fixed_path)
            os.replace(fixed_path, file_path)
            
            # Count lines again
            line_count = 0
            with open(file_path, "r") as f:
                for line in f:
                    json.loads(line)
                    line_count += 1
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid JSONL format and couldn't fix: {str(e)}")
    
    return {
        "filename": file.filename,
        "size": os.path.getsize(file_path),
        "samples": line_count
    }


def fix_jsonl(input_filename, output_filename):
    """Fix JSONL file with improperly escaped newlines"""
    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        for line in infile:
            try:
                json.loads(line)
                outfile.write(line)
            except json.JSONDecodeError:
                fixed_line = line.replace("\\n", "\\\\n").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                try:
                    json.loads(fixed_line)
                    outfile.write(fixed_line + "\n")
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not fix line: {line[:50]}... Error: {e}")


@app.get("/datasets")
async def list_datasets():
    """List all available datasets"""
    datasets = []
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(DATASET_DIR, filename)
            
            # Count samples
            try:
                line_count = 0
                with open(file_path, "r") as f:
                    for _ in f:
                        line_count += 1
            except Exception:
                line_count = -1
            
            datasets.append({
                "name": filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "samples": line_count,
                "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            })
    return datasets


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Delete a dataset"""
    dataset_path = os.path.join(DATASET_DIR, name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset {name} not found")
    
    try:
        os.remove(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")
    
    return {"message": f"Dataset {name} deleted"}


# Training Management
async def run_training_job(job_id: str, adapter_config: AdapterConfig):
    """Background task to run a training job"""
    try:
        # Ensure Docker network exists
        ensure_docker_network()
        
        # Update job status
        active_jobs[job_id]["status"] = "starting"
        
        # Load model config
        model_config_path = os.path.join(CONFIG_DIR, f"{adapter_config.base_model}.toml")
        if not os.path.exists(model_config_path):
            raise Exception(f"Model config {adapter_config.base_model} not found")
        
        model_config = toml.load(model_config_path)
        
        # Prepare adapter output directory
        adapter_output_dir = os.path.join(ADAPTER_DIR, adapter_config.name)
        os.makedirs(adapter_output_dir, exist_ok=True)
        
        # Prepare log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"training_{adapter_config.name}_{timestamp}.log")
        
        # Build training command with all options from config
        # This directly builds a python command instead of relying on entrypoint scripts
        cmd = [
            "python", "/workspace/train.py",
            "--model-name", model_config["model_id"],
            "--dataset", f"/workspace/training/{os.path.basename(adapter_config.dataset)}",
            "--output-dir", f"/workspace/adapters/{adapter_config.name}"
        ]
        
        # Add all non-None parameters from the adapter config
        config_dict = adapter_config.model_dump(exclude={"name", "description", "base_model", "dataset"})
        for key, value in config_dict.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Add log file option
        cmd.extend(["--log-file", f"/workspace/logs/{os.path.basename(log_file)}"])

        # Launch the training container with proper volume mounts
        container = docker_client.containers.run(
            "llora-lab-trainer",
            command=cmd,
            volumes={
                os.path.abspath(DATASET_DIR): {"bind": "/workspace/training", "mode": "ro"},
                os.path.abspath(ADAPTER_DIR): {"bind": "/workspace/adapters", "mode": "rw"},
                os.path.abspath(LOG_DIR): {"bind": "/workspace/logs", "mode": "rw"},
                os.path.abspath(HF_CACHE_DIR): {"bind": "/huggingface-cache", "mode": "rw"}
            },
            environment={
                "HF_HOME": "/huggingface-cache",
                "PYTHONUNBUFFERED": "1",
                "HF_TOKEN": HF_TOKEN
            },
            detach=True,
            remove=True,
            runtime="nvidia",
            network=DOCKER_NETWORK_NAME
        )
        
        # Update job with container ID
        active_jobs[job_id]["container_id"] = container.id
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["log_file"] = log_file
        
        # Monitor the training progress by reading the log file
        logger.info(f"Training job {job_id} started, container ID: {container.id}")
        
        # Wait for log file to be created
        start_time = datetime.now()
        max_wait_seconds = 60
        log_file_created = False
        
        while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
            # Check if container is still running
            try:
                container.reload()
                if container.status != "running":
                    active_jobs[job_id]["status"] = "failed"
                    active_jobs[job_id]["message"] = "Container stopped unexpectedly"
                    return
            except docker.errors.NotFound:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["message"] = "Container not found"
                return
                
            # Check if log file exists
            if os.path.exists(log_file):
                log_file_created = True
                break
                
            await asyncio.sleep(1)
        
        if not log_file_created:
            # If log file wasn't created, use container logs instead
            active_jobs[job_id]["message"] = "No log file created, using container logs"
            active_jobs[job_id]["using_container_logs"] = True
        
        # Monitor job progress
        while True:
            # Keep container status updated
            try:
                container.reload()
                if container.status != "running":
                    break
            except docker.errors.NotFound:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["message"] = "Container not found"
                break
            
            # Read logs - either from file or container
            if log_file_created:
                # Read new lines from log file
                try:
                    with open(log_file, "r") as f:
                        log_content = f.read()
                    parse_training_log(job_id, log_content)
                except Exception as e:
                    logger.error(f"Error reading log file: {str(e)}")
            else:
                # Read from container logs
                try:
                    logs = container.logs(since=active_jobs[job_id].get("last_log_timestamp", 0)).decode("utf-8")
                    if logs:
                        parse_training_log(job_id, logs)
                        active_jobs[job_id]["last_log_timestamp"] = int(datetime.now().timestamp())
                except Exception as e:
                    logger.error(f"Error reading container logs: {str(e)}")
            
            await asyncio.sleep(2)
            
        # Container has stopped, check final status
        try:
            # Force a final update from any remaining logs
            if log_file_created and os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_content = f.read()
                parse_training_log(job_id, log_content)
            else:
                logs = container.logs().decode("utf-8")
                parse_training_log(job_id, logs)
            
            # Check if training completed successfully
            if active_jobs[job_id]["status"] != "completed":
                if os.path.exists(os.path.join(adapter_output_dir, "adapter_model.bin")):
                    active_jobs[job_id]["status"] = "completed"
                    active_jobs[job_id]["message"] = "Training completed successfully"
                else:
                    active_jobs[job_id]["status"] = "failed"
                    active_jobs[job_id]["message"] = "Training job stopped without completing"
        except Exception as e:
            logger.error(f"Error parsing final logs: {str(e)}")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = str(e)
    
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}", exc_info=True)
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = str(e)


def parse_training_log(job_id: str, log_content: str):
    """Parse training logs to update job status"""
    try:
        # Extract training step, loss, etc.
        job = active_jobs[job_id]
        
        # Check for step information
        for line in log_content.split('\n'):
            if "step" in line.lower() and "loss" in line.lower():
                # Look for patterns like "Step 10/60, loss: 2.345, lr: 0.0001"
                parts = line.split()
                
                # Find step information
                for i, part in enumerate(parts):
                    if "step" in part.lower() and i + 1 < len(parts):
                        step_info = parts[i + 1].strip(',')
                        if '/' in step_info:
                            current_step, total_steps = map(int, step_info.split('/'))
                            job["step"] = current_step
                            job["total_steps"] = total_steps
                
                # Find loss information
                for i, part in enumerate(parts):
                    if "loss:" in part.lower() and i + 1 < len(parts):
                        try:
                            loss_value = float(parts[i + 1].strip(','))
                            job["loss"] = loss_value
                        except ValueError:
                            pass
                
                # Find learning rate information
                for i, part in enumerate(parts):
                    if "lr:" in part.lower() and i + 1 < len(parts):
                        try:
                            lr_value = float(parts[i + 1].strip(','))
                            job["learning_rate"] = lr_value
                        except ValueError:
                            pass
        
        # Check for completion
        if "training complete" in log_content.lower() or "model saved" in log_content.lower():
            job["status"] = "completed"
            job["message"] = "Training completed successfully"
        
    except Exception as e:
        logger.error(f"Error parsing log: {str(e)}")


@app.post("/training/start", response_model=TrainingJob)
async def start_training_job(adapter_name: str, background_tasks: BackgroundTasks):
    """Start a training job for an adapter configuration"""
    # Load adapter config
    config_path = os.path.join(CONFIG_DIR, f"adapter_{adapter_name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Adapter config {adapter_name} not found")
    
    adapter_config = toml.load(config_path)
    adapter_config["name"] = adapter_name
    adapter_config = AdapterConfig(**adapter_config)
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job with defaults 
    # Note: We don't hardcode step counts or other values, these come from the config
    total_steps = adapter_config.steps or 60  # Use config value or fallback to default
    
    job = {
        "id": job_id,
        "adapter_config": adapter_name,
        "status": "initializing",
        "start_time": datetime.now().isoformat(),
        "step": 0,
        "total_steps": total_steps,
        "loss": None,
        "learning_rate": None,
        "message": None
    }
    
    # Add to active jobs
    active_jobs[job_id] = job
    
    # Start background task
    background_tasks.add_task(run_training_job, job_id, adapter_config)
    
    return TrainingJob(**job)


@app.get("/training/jobs", response_model=List[TrainingJob])
async def list_training_jobs():
    """List all training jobs"""
    jobs = []
    for job_id, job in active_jobs.items():
        if "adapter_config" in job:  # Only include training jobs
            jobs.append(TrainingJob(**job))
    return jobs


@app.get("/training/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """Get details of a specific training job"""
    if job_id not in active_jobs or "adapter_config" not in active_jobs[job_id]:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    return TrainingJob(**active_jobs[job_id])


@app.get("/training/logs/{job_id}")
async def get_training_logs(job_id: str, lines: int = 100):
    """Get logs for a specific training job"""
    if job_id not in active_jobs or "adapter_config" not in active_jobs[job_id]:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job = active_jobs[job_id]
    
    # First check if we're using container logs
    if job.get("using_container_logs", False) and "container_id" in job:
        try:
            container = docker_client.containers.get(job["container_id"])
            logs = container.logs(tail=lines).decode("utf-8")
            return {"logs": format_container_logs(logs)}
        except Exception as e:
            logger.error(f"Error getting container logs: {str(e)}")
            return {"logs": f"Error reading logs: {str(e)}"}
    
    # Otherwise try to read log file
    if "log_file" not in job or not os.path.exists(job["log_file"]):
        return {"logs": "No logs available"}
    
    # Read the last N lines from the log file
    try:
        with open(job["log_file"], "r") as f:
            all_lines = f.readlines()
            
        # Return the last N lines
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {"logs": "".join(log_lines)}
    except Exception as e:
        logger.error(f"Error reading log file: {str(e)}")
        return {"logs": f"Error reading logs: {str(e)}"}


@app.delete("/training/jobs/{job_id}")
async def stop_training_job(job_id: str):
    """Stop a training job"""
    if job_id not in active_jobs or "adapter_config" not in active_jobs[job_id]:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    job = active_jobs[job_id]
    
    if "container_id" in job:
        try:
            container = docker_client.containers.get(job["container_id"])
            container.stop()
            job["status"] = "stopped"
            job["message"] = "Training job was manually stopped"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error stopping container: {str(e)}")
    else:
        job["status"] = "stopped"
        job["message"] = "Training job was manually stopped"
    
    return {"message": f"Training job {job_id} stopped"}


# Adapters Management
@app.get("/adapters")
async def list_adapters():
    """List all available adapters"""
    adapters = []
    for dirname in os.listdir(ADAPTER_DIR):
        adapter_path = os.path.join(ADAPTER_DIR, dirname)
        if os.path.isdir(adapter_path):
            # Check for adapter config
            config_path = os.path.join(CONFIG_DIR, f"adapter_{dirname}.toml")
            base_model = None
            if os.path.exists(config_path):
                try:
                    config = toml.load(config_path)
                    base_model = config.get("base_model")
                except Exception:
                    pass
            
            adapters.append({
                "name": dirname,
                "path": adapter_path,
                "size": sum(os.path.getsize(os.path.join(adapter_path, f)) for f in os.listdir(adapter_path) if os.path.isfile(os.path.join(adapter_path, f))),
                "created": datetime.fromtimestamp(os.path.getctime(adapter_path)).isoformat(),
                "base_model": base_model
            })
    return adapters


@app.delete("/adapters/{name}")
async def delete_adapter(name: str):
    """Delete an adapter"""
    adapter_path = os.path.join(ADAPTER_DIR, name)
    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail=f"Adapter {name} not found")
    
    try:
        shutil.rmtree(adapter_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting adapter: {str(e)}")
    
    # Also delete the config if it exists
    config_path = os.path.join(CONFIG_DIR, f"adapter_{name}.toml")
    if os.path.exists(config_path):
        os.remove(config_path)
    
    return {"message": f"Adapter {name} deleted"}


def get_host_mount_paths():
    """Get the host paths for all mounted volumes in current container"""
    try:
        # Get current container ID
        container_id = os.environ.get('HOSTNAME', '')
        
        # Inspect the container to get mount information
        inspect_data = docker_client.api.inspect_container(container_id)
        
        # Map of container paths to host paths
        mount_map = {}
        
        # Find all mounts
        for mount in inspect_data['Mounts']:
            mount_map[mount['Destination']] = mount['Source']
        
        return mount_map
    except Exception as e:
        logger.error(f"Error getting host mount paths: {str(e)}")
        return {}

# Serving Management
async def run_serving_job(job_id: str, model_config: ModelConfig, adapter: Optional[str] = None):
    """Background task to run a serving job"""
    try:
        # Ensure Docker network exists
        ensure_docker_network()
        
        # Update job status
        active_jobs[job_id]["status"] = "starting"
        active_jobs[job_id]["message"] = "Preparing to start vLLM server..."
        
        # Prepare log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"serving_{model_config.name}_{timestamp}.log")
        active_jobs[job_id]["log_file"] = log_file
        
        # Stop any existing vLLM or webui containers
        stop_existing_containers()
        
        # Build the vLLM command
        cmd = ["vllm", "serve", model_config.model_id, "--host", "0.0.0.0", "--port", "8000"]
        
        # Add all non-None parameters from the model config
        config_dict = model_config.model_dump(exclude={"name", "description", "model_id", "additional_params"})
        for key, value in config_dict.items():
            if value is not None:
                # Convert to command line format (snake_case to kebab-case)
                param_name = f"--{key.replace('_', '-')}"
                
                # Handle boolean flags specially
                if isinstance(value, bool):
                    if value:
                        cmd.append(param_name)
                else:
                    cmd.append(param_name)
                    cmd.append(str(value))
        
        # Add adapter config if specified
        if adapter:
            cmd.append("--enable-lora")
            cmd.append("--lora-modules")
            cmd.append(f"latest=/adapters/{adapter}")
        
        # Add additional parameters from config
        for key, value in model_config.additional_params.items():
            # Convert to command line format
            param_name = f"--{key.replace('_', '-')}"
            
            # Handle boolean flags specially
            if isinstance(value, bool):
                if value:
                    cmd.append(param_name)
            else:
                cmd.append(param_name)
                cmd.append(str(value))
        
        # Update status message with command
        active_jobs[job_id]["message"] = f"Starting vLLM with command: {' '.join(cmd)}"
        logger.info(f"Starting vLLM with command: {' '.join(cmd)}")


        logger.info('USING ' + str(HF_CACHE_DIR))
        
        # Launch the vLLM server container
        environment = {
            "HF_HOME": "/huggingface-cache",
            "PYTHONUNBUFFERED": "1",
            "HF_TOKEN": HF_TOKEN
        }
        
        # Get host mount paths
        host_mounts = get_host_mount_paths()
        
        # The container paths in the admin-api container
        hf_cache_container_path = '/app/huggingface-cache'
        adapters_container_path = '/app/adapters'
        logs_container_path = '/app/logs'

        # Get corresponding host paths
        hf_cache_host_path = host_mounts.get(hf_cache_container_path)
        adapters_host_path = host_mounts.get(adapters_container_path)
        logs_host_path = host_mounts.get(logs_container_path)

        # Make sure we have all paths
        if not all([hf_cache_host_path, adapters_host_path, logs_host_path]):
            raise Exception(f"Could not determine host paths: {host_mounts}")
        
        # Update job with paths
        active_jobs[job_id]["status"] = "starting"
        active_jobs[job_id]["message"] = f"Determined mount paths. Starting vLLM..."

        container = docker_client.containers.run(
            "llora-lab-vllm",
            command=cmd,
            volumes={
                adapters_host_path: {"bind": "/adapters", "mode": "ro"},
                logs_host_path: {"bind": "/logs", "mode": "rw"},
                hf_cache_host_path: {"bind": "/huggingface-cache", "mode": "rw"}
            },
            environment=environment,
            ports={"8000/tcp": 8000},
            detach=True,
            remove=True,
            runtime="nvidia",
            name="llora-lab-vllm",
            ipc_mode="host",
            ulimits=[
                docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
                docker.types.Ulimit(name="stack", soft=67108864, hard=67108864)
            ],
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,
                    capabilities=[['gpu']]
                )
            ],
            network=DOCKER_NETWORK_NAME
        )
        
        # Check if the network connection is working
        active_jobs[job_id]["message"] = "Container started, verifying network..."
        time.sleep(3)  # Give the container a moment to fully start

        # Test network connectivity using docker exec
        try:
            # Run a network test inside the container to check connectivity
            network_test = docker_client.containers.run(
                "busybox",
                command=["ping", "-c", "2", "llora-lab-vllm"],
                network=DOCKER_NETWORK_NAME,
                remove=True
            )
            logger.info(f"Network test result: {network_test.decode('utf-8')}")
            active_jobs[job_id]["message"] = "Network connectivity verified"
        except Exception as e:
            logger.warning(f"Network test failed: {str(e)}")
            active_jobs[job_id]["message"] = f"Network warning: {str(e)}"
        
        # Update job with container ID
        active_jobs[job_id]["container_id"] = container.id
        active_jobs[job_id]["status"] = "initializing"
        active_jobs[job_id]["message"] = "vLLM container started, waiting for server to initialize..."
        
        logger.info(f"vLLM container started with ID: {container.id}")
        
        # Wait for server to be ready
        server_ready = False
        start_time = datetime.now()
        timeout_seconds = 300  # 5 minutes timeout
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            try:
                container.reload()
                if container.status != "running":
                    raise Exception(f"Container stopped with status: {container.status}")
                
                # Check logs for server ready message
                logs = container.logs().decode("utf-8")
                
                if "Application startup complete" in logs or "Server is ready" in logs:
                    server_ready = True
                    active_jobs[job_id]["status"] = "running"
                    active_jobs[job_id]["message"] = "vLLM server is running, starting web UI..."
                    break
                
                # Update status with progress information
                if "Loading model" in logs:
                    active_jobs[job_id]["message"] = "Loading model weights..."
                elif "Compiling model" in logs:
                    active_jobs[job_id]["message"] = "Compiling model..."
                
            except Exception as e:
                logger.error(f"Error checking vLLM container: {str(e)}")
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["message"] = f"vLLM server failed to start: {str(e)}"
                return
            
            await asyncio.sleep(5)
        
        if not server_ready:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = "vLLM server failed to start within timeout period"
            container.stop()
            return
        
        # Start the OpenWebUI container
        try:
            webui_container = docker_client.containers.run(
                "ghcr.io/open-webui/open-webui:main",
                environment={
                    "OPENAI_API_BASE_URL": "http://llora-lab-vllm:8000/v1"
                },
                ports={"8080/tcp": 3000},
                volumes={
                    "webui-data": {"bind": "/app/backend/data", "mode": "rw"}
                },
                detach=True,
                remove=True,
                name="llora-lab-webui",
                network=DOCKER_NETWORK_NAME
            )
            
            active_jobs[job_id]["webui_container_id"] = webui_container.id
            active_jobs[job_id]["status"] = "ready"
            active_jobs[job_id]["message"] = "Model is now ready to serve requests via OpenWebUI and API"
            logger.info(f"Started OpenWebUI with container ID: {webui_container.id}")
            
            # Test the API to confirm it's working
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("http://llora-lab-vllm:8000/v1/models")
                    if response.status_code == 200:
                        logger.info("API is working correctly" + response.text)
                    else:
                        logger.warning(f"API returned status code: {response.status_code}")
            except Exception as e:
                logger.warning(f"Could not test API: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to start OpenWebUI: {str(e)}")
            active_jobs[job_id]["message"] = f"vLLM server is ready, but WebUI failed to start: {str(e)}"
        
        # Keep containers running until manually stopped
        while True:
            try:
                # First check vLLM container
                container.reload()
                if container.status != "running":
                    active_jobs[job_id]["status"] = "stopped"
                    active_jobs[job_id]["message"] = "vLLM server stopped unexpectedly"
                    break
                
                # Then check WebUI container if it exists
                if "webui_container_id" in active_jobs[job_id]:
                    try:
                        webui_container = docker_client.containers.get(active_jobs[job_id]["webui_container_id"])
                        webui_container.reload()
                        if webui_container.status != "running":
                            logger.warning("WebUI container stopped unexpectedly")
                            active_jobs[job_id]["message"] = "WebUI stopped unexpectedly, but vLLM server is still running"
                            # Remove the container ID so we don't keep checking it
                            active_jobs[job_id].pop("webui_container_id", None)
                    except docker.errors.NotFound:
                        logger.warning("WebUI container not found")
                        active_jobs[job_id]["message"] = "WebUI container not found, but vLLM server is still running"
                        active_jobs[job_id].pop("webui_container_id", None)
                    except Exception as e:
                        logger.error(f"Error checking WebUI container: {str(e)}")
                
                # Update request statistics (if available)
                # This would normally come from the vLLM API metrics
                
            except docker.errors.NotFound:
                active_jobs[job_id]["status"] = "stopped"
                active_jobs[job_id]["message"] = "vLLM container not found"
                break
            except Exception as e:
                logger.error(f"Error monitoring containers: {str(e)}")
                active_jobs[job_id]["message"] = f"Error monitoring containers: {str(e)}"
            
            await asyncio.sleep(5)
    
    except Exception as e:
        logger.error(f"Serving job {job_id} failed: {str(e)}", exc_info=True)
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = str(e)
        
        # Clean up any containers that might be running
        stop_existing_containers()


def stop_existing_containers():
    """Stop any existing vLLM or webui containers"""
    for container_name in ["llora-lab-vllm", "llora-lab-webui"]:
        try:
            container = docker_client.containers.get(container_name)
            logger.info(f"Stopping existing container: {container_name}")
            container.stop()
        except docker.errors.NotFound:
            pass  # Container doesn't exist, which is fine
        except Exception as e:
            logger.error(f"Error stopping container {container_name}: {str(e)}")


@app.post("/serving/start", response_model=ServingJob)
async def start_serving_job(model_name: str, background_tasks: BackgroundTasks, adapter: Optional[str] = None):
    """Start a serving job for a model configuration with optional adapter"""
    # Check for existing serving jobs
    for job_id, job in active_jobs.items():
        if "model_conf" in job and job.get("status") in ["starting", "initializing", "running", "ready"]:
            raise HTTPException(
                status_code=400, 
                detail="A serving job is already running. Stop it before starting a new one."
            )
    
    # Load model config
    config_path = os.path.join(CONFIG_DIR, f"{model_name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Model config {model_name} not found")
    
    model_config = toml.load(config_path)
    model_config["name"] = model_name
    model_config = ModelConfig(**model_config)
    
    # Check adapter if specified
    if adapter:
        adapter_path = os.path.join(ADAPTER_DIR, adapter)
        if not os.path.exists(adapter_path):
            raise HTTPException(status_code=404, detail=f"Adapter {adapter} not found")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job
    job = {
        "id": job_id,
        "model_conf": model_name,
        "model_id": model_config.model_id,
        "adapter": adapter,
        "status": "initializing",
        "start_time": datetime.now().isoformat(),
        "requests_served": 0,
        "avg_response_time": 0.0,
        "message": "Setting up serving job..."
    }
    
    # Add to active jobs
    active_jobs[job_id] = job
    
    # Start background task
    background_tasks.add_task(run_serving_job, job_id, model_config, adapter)
    
    return ServingJob(**job)


@app.get("/serving/jobs", response_model=List[ServingJob])
async def list_serving_jobs():
    """List all serving jobs"""
    jobs = []
    for job_id, job in active_jobs.items():
        if "model_conf" in job:  # Only include serving jobs
            jobs.append(ServingJob(**job))
    return jobs


@app.get("/serving/jobs/{job_id}", response_model=ServingJob)
async def get_serving_job(job_id: str):
    """Get details of a specific serving job"""
    if job_id not in active_jobs or "model_conf" not in active_jobs[job_id]:
        raise HTTPException(status_code=404, detail=f"Serving job {job_id} not found")
    
    return ServingJob(**active_jobs[job_id])


@app.get("/serving/logs/{job_id}")
async def get_serving_logs(job_id: str, lines: int = 100):
    """Get logs for a specific serving job"""
    if job_id not in active_jobs or "model_conf" not in active_jobs[job_id]:
        raise HTTPException(status_code=404, detail=f"Serving job {job_id} not found")
    
    job = active_jobs[job_id]
    
    # Try to get logs from container directly
    if "container_id" in job:
        try:
            container = docker_client.containers.get(job["container_id"])
            logs = container.logs(tail=lines).decode("utf-8")
            return {"logs": format_container_logs(logs)}
        except Exception as e:
            logger.error(f"Error getting container logs: {str(e)}")
            return {"logs": f"Error getting logs: {str(e)}"}
    
    # Fallback to log file if available
    if "log_file" in job and os.path.exists(job["log_file"]):
        try:
            with open(job["log_file"], "r") as f:
                all_lines = f.readlines()
                
            # Return the last N lines
            log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return {"logs": "".join(log_lines)}
        except Exception as e:
            logger.error(f"Error reading log file: {str(e)}")
            return {"logs": f"Error reading logs: {str(e)}"}
    
    return {"logs": "No logs available"}


@app.delete("/serving/jobs/{job_id}")
async def stop_serving_job(job_id: str):
    """Stop a serving job"""
    if job_id not in active_jobs or "model_conf" not in active_jobs[job_id]:
        raise HTTPException(status_code=404, detail=f"Serving job {job_id} not found")
    
    job = active_jobs[job_id]
    
    # Stop WebUI container if it exists
    if "webui_container_id" in job:
        try:
            webui_container = docker_client.containers.get(job["webui_container_id"])
            webui_container.stop()
            logger.info(f"Stopped WebUI container: {job['webui_container_id']}")
        except Exception as e:
            logger.error(f"Error stopping WebUI container: {str(e)}")
    
    # Stop vLLM container
    if "container_id" in job:
        try:
            container = docker_client.containers.get(job["container_id"])
            container.stop()
            job["status"] = "stopped"
            job["message"] = "Serving job was manually stopped"
            logger.info(f"Stopped vLLM container: {job['container_id']}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error stopping container: {str(e)}")
    else:
        job["status"] = "stopped"
        job["message"] = "Serving job was manually stopped"
    
    return {"message": f"Serving job {job_id} stopped"}


# Testing endpoints
@app.post("/test/model")
async def test_model(prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256):
    """Test the currently active model using the OpenAI API endpoint"""
    # Find active serving job
    active_job = None
    for job_id, job in active_jobs.items():
        if "model_conf" in job and job.get("status") == "ready":
            active_job = job
            break
    
    logger.info(f"Model is: {str(job)}")

    if not active_job:
        raise HTTPException(status_code=400, detail="No active model is currently being served")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            api_url = "http://llora-lab-vllm:8000/v1/completions"
            
            payload = {
                "model": job.get("model_id"),
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
            
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Update job stats
            active_job["requests_served"] = active_job.get("requests_served", 0) + 1
            
            # Extract response
            content = result["choices"][0]["text"]
            
            return {
                "prompt": prompt,
                "response": content
            }
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error communicating with model API: {str(e)}"
        )


@app.post("/test/adapter")
async def test_adapter(adapter_name: str, prompt: str, 
                      temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256):
    """Test an adapter with a given prompt"""
    adapter_path = os.path.join(ADAPTER_DIR, adapter_name)
    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail=f"Adapter {adapter_name} not found")
    
    # Check if the adapter is already being served
    for job in active_jobs.values():
        if "adapter" in job and job.get("adapter") == adapter_name and job.get("status") == "ready":
            # Adapter is already loaded in a running vLLM instance
            # Make a direct API call to it
            try:
                # Just use the standard model test endpoint
                result = await test_model(prompt, temperature, top_p, max_tokens)
                result["adapter"] = adapter_name
                return result
            except Exception as e:
                logger.error(f"Error testing adapter via API: {str(e)}")
                # Fall back to the container method below
    
    # Load adapter config to get base model
    config_path = os.path.join(CONFIG_DIR, f"adapter_{adapter_name}.toml")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Adapter config for {adapter_name} not found")
    
    adapter_config = toml.load(config_path)
    base_model = adapter_config.get("base_model")
    
    if not base_model:
        raise HTTPException(status_code=400, detail=f"Base model not specified in adapter config")
    
    # Load model config
    model_config_path = os.path.join(CONFIG_DIR, f"{base_model}.toml")
    if not os.path.exists(model_config_path):
        raise HTTPException(status_code=404, detail=f"Model config {base_model} not found")
    
    model_config = toml.load(model_config_path)
    
    # Run test using the train.py script directly with test mode
    try:
        # Ensure Docker network exists
        ensure_docker_network()
        
        # Track unique session to avoid conflicts
        test_id = uuid.uuid4().hex[:8]
        log_file = os.path.join(LOG_DIR, f"adapter_test_{test_id}.log")
        
        cmd = [
            "python", "/workspace/test_adapter.py",
            "--adapter", f"/workspace/adapters/{adapter_name}",
            "--prompt", prompt,
            "--temperature", str(temperature),
            "--top-p", str(top_p),
            "--max-tokens", str(max_tokens),
            "--log-level", "info"
        ]
        
        # Run the container in interactive mode to get output
        container = docker_client.containers.run(
            "llora-lab-trainer",
            command=cmd,
            volumes={
                os.path.abspath(ADAPTER_DIR): {"bind": "/workspace/adapters", "mode": "ro"},
                os.path.abspath(LOG_DIR): {"bind": "/workspace/logs", "mode": "rw"},
                os.path.abspath(HF_CACHE_DIR): {"bind": "/huggingface-cache", "mode": "rw"}
            },
            environment={
                "HF_HOME": "/huggingface-cache",
                "PYTHONUNBUFFERED": "1",
                "HF_TOKEN": HF_TOKEN,
                "SAVE_RESPONSE": "1"  # Tell the script to save response to file
            },
            remove=True,
            runtime="nvidia",
            network=DOCKER_NETWORK_NAME
        )
        
        output = container.decode("utf-8")
        
        # Extract response from output
        response_text = ""
        capture = False
        for line in output.split("\n"):
            if "Response:" in line:
                capture = True
                continue
            if capture:
                response_text += line + "\n"
        
        return {
            "adapter": adapter_name,
            "prompt": prompt,
            "response": response_text.strip()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing adapter: {str(e)}")


# System Endpoints
@app.get("/system/stats")
@cache(expire=10)
async def get_system_stats():
    """Get system statistics including GPU usage"""
    try:
        # Docker stats
        containers = docker_client.containers.list()
        container_stats = []
        for container in containers:
            try:
                stats = container.stats(stream=False)
                container_stats.append({
                    "id": container.id[:12],
                    "name": container.name,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else container.image.id[:12],
                    "cpu_percent": calculate_cpu_percent(stats),
                    "memory_usage": stats["memory_stats"].get("usage", 0),
                    "memory_limit": stats["memory_stats"].get("limit", 0)
                })
            except Exception as e:
                logger.error(f"Error getting stats for container {container.id}: {str(e)}")
        
        # GPU stats (using nvidia-smi)
        gpu_stats = latest_gpu_stats
        
        return {
            "containers": container_stats,
            "gpu": gpu_stats,
            "active_jobs": len(active_jobs),
            "disk_usage": get_disk_usage()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")


@app.get("/system/metrics")
@cache(expire=10)
async def get_system_metrics():
    """Get detailed system metrics for monitoring dashboard"""
    try:
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Get GPU utilization
        gpu_stats = latest_gpu_stats
        gpu_util = 0.0
        if isinstance(gpu_stats, dict) and "utilized" in gpu_stats:
            try:
                gpu_util = float(gpu_stats["utilized"])
            except:
                pass
                
        # Get memory usage
        memory_stats = {}
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.readlines()
                
            memory_data = {}
            for line in meminfo:
                if ":" in line:
                    key, value = line.split(":", 1)
                    memory_data[key.strip()] = value.strip()
                    
            if "MemTotal" in memory_data and "MemAvailable" in memory_data:
                total = int(memory_data["MemTotal"].split()[0])
                available = int(memory_data["MemAvailable"].split()[0])
                used = total - available
                memory_stats = {
                    "total": total,
                    "used": used,
                    "available": available,
                    "percent_used": (used / total) * 100
                }
        except Exception as e:
            logger.error(f"Error getting memory info: {str(e)}")
            memory_stats = {"error": str(e)}
        
        return {
            "timestamp": timestamp,
            "gpu_utilization": gpu_util,
            "memory_usage": memory_stats,
            "disk_usage": get_disk_usage(),
            "active_jobs_count": len(active_jobs)
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")


# def calculate_cpu_percent(stats):
#     """Calculate CPU percentage from Docker stats"""
#     cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
#     system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
    
#     if system_delta > 0 and cpu_delta > 0:
#         cpu_count = len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
#         return (cpu_delta / system_delta) * cpu_count * 100.0
    
#     return 0.0

def calculate_cpu_percent(stats):
    """Calculate CPU percentage from Docker stats"""
    try:
        # First check if we have all required fields
        if not all(key in stats for key in ["cpu_stats", "precpu_stats"]):
            return 0.0
            
        # Determine CPU count safely
        cpu_count = 1
        if "percpu_usage" in stats["cpu_stats"]["cpu_usage"]:
            cpu_count = len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
        elif "online_cpus" in stats["cpu_stats"]:
            cpu_count = stats["cpu_stats"]["online_cpus"]
        
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
        
        if system_delta > 0 and cpu_delta > 0:
            return (cpu_delta / system_delta) * cpu_count * 100.0
        
        return 0.0
    except Exception as e:
        logger.debug(f"Error calculating CPU percent: {str(e)}")
        return 0.0

async def update_gpu_stats_periodically():
    """Background task that updates GPU stats every 15 seconds"""
    while True:
        try:
            # Get stats using your existing container method
            stats = await get_gpu_stats_internal()
            
            # Update the shared variable with timestamp
            stats["last_updated"] = time.time()
            global latest_gpu_stats
            latest_gpu_stats = stats
            
            # Log success
            logger.info("Updated GPU stats successfully")
            
        except Exception as e:
            logger.error(f"Failed to update GPU stats: {str(e)}")
        
        # Wait 15 seconds before next update
        await asyncio.sleep(15)

@app.on_event("startup")
async def start_background_tasks():
    # Start background task without awaiting it
    asyncio.create_task(update_gpu_stats_periodically())


async def get_gpu_stats_internal():
    """Get GPU statistics using nvidia-smi"""
    try:
        # Run nvidia-smi in a container for consistent access
        container = docker_client.containers.run(
            "nvidia/cuda:12.0.0-base-ubuntu22.04",
            command=["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", 
                    "--format=csv,noheader,nounits"],
            remove=True,
            runtime="nvidia"
        )
        
        output = container.decode("utf-8").strip()
        
        # Parse the CSV output
        gpus = []
        for line in output.split("\n"):
            if not line.strip():
                continue
                
            parts = line.split(",")
            if len(parts) >= 6:
                index, name, mem_used, mem_total, util, temp = [p.strip() for p in parts[:6]]
                gpus.append({
                    "index": index,
                    "name": name,
                    "memory": f"{mem_used}/{mem_total} MB",
                    "utilization": f"{util}%",
                    "temperature": f"{temp}°C"
                })
        
        # If we have GPU info, calculate memory utilization
        if gpus:
            first_gpu = gpus[0]
            mem_parts = first_gpu["memory"].split("/")
            if len(mem_parts) == 2:
                try:
                    used_mem = float(mem_parts[0].strip())
                    total_mem = float(mem_parts[1].split()[0].strip())
                    utilized = used_mem / total_mem
                except:
                    utilized = 0
            else:
                utilized = 0
                
            return {
                "gpus": gpus,
                "utilized": f"{utilized:.2f}",
                "temperature": first_gpu["temperature"],
                "memory": first_gpu["memory"]
            }
        
        return {"error": "No GPU information available"}
        
    except Exception as e:
        logger.error(f"Error getting GPU stats: {str(e)}")
        return {"error": str(e)}


# def get_disk_usage():
#     """Get disk usage for important directories"""
#     usage = {}
    
#     for name, path in [
#         ("configs", CONFIG_DIR),
#         ("datasets", DATASET_DIR),
#         ("adapters", ADAPTER_DIR),
#         ("logs", LOG_DIR),
#         ("huggingface_cache", HF_CACHE_DIR)
#     ]:
#         try:
#             total_size = 0
#             for dirpath, _, filenames in os.walk(path):
#                 for f in filenames:
#                     fp = os.path.join(dirpath, f)
#                     if os.path.exists(fp):
#                         total_size += os.path.getsize(fp)
            
#             usage[name] = {
#                 "size_bytes": total_size,
#                 "size_human": format_size(total_size)
#             }
#         except Exception as e:
#             logger.error(f"Error calculating size for {name}: {str(e)}")
#             usage[name] = {"error": str(e)}
    
#     return usage

def get_disk_usage():
    usage = {}
    
    for name, path in [
        ("configs", CONFIG_DIR),
        ("datasets", DATASET_DIR),
        ("adapters", ADAPTER_DIR),
        ("logs", LOG_DIR),
        ("huggingface_cache", HF_CACHE_DIR)
    ]:
        try:
            # Use du command which is much faster than Python traversal
            import subprocess
            result = subprocess.run(
                ["du", "-sb", path],
                capture_output=True, text=True, timeout=5
            ).stdout
            
            if result:
                size_str = result.split()[0]
                total_size = int(size_str)
                
                usage[name] = {
                    "size_bytes": total_size,
                    "size_human": format_size(total_size)
                }
        except Exception as e:
            logger.error(f"Error calculating size for {name}: {str(e)}")
            usage[name] = {"error": str(e)}
    
    return usage


def format_size(size_bytes):
    """Format size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0


# Initialize with default configs
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    # Create default model configs if they don't exist
    default_models = [
        {
            "name": "llama-3.1-8b",
            "description": "Meta Llama 3.1 8B Instruct",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.99,
            "additional_params": {
                "chat_template": "llama-3.1",
                "response_role": "assistant"
            }
        },
        {
            "name": "llama-3-70b",
            "description": "Meta Llama 3 70B Instruct",
            "model_id": "meta-llama/Llama-3-70B-Instruct",
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.99,
            "tensor_parallel_size": 2,  # Set this for multi-GPU by default
            "additional_params": {
                "chat_template": "llama-3",
                "response_role": "assistant"
            }
        },
        {
            "name": "llama-3.1-vision",
            "description": "Meta Llama 3.1 8B Vision",
            "model_id": "meta-llama/Llama-3.1-8B-Vision",
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
            "max_model_len": 4096,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.9,
            "additional_params": {
                "chat_template": "llama-3.1",
                "response_role": "assistant"
            }
        },
        {
            "name": "mistral-7b",
            "description": "Mistral 7B Instruct v0.2",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "quantization": "bitsandbytes", 
            "load_format": "bitsandbytes",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.99,
            "additional_params": {
                "chat_template": "mistral",
                "response_role": "assistant"
            }
        }
    ]
    
    for model in default_models:
        config_path = os.path.join(CONFIG_DIR, f"{model['name']}.toml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                toml.dump(model, f)
    
    logger.info("Llora Lab API initialized with default configurations")
    
    # Check for docker socket
    if not os.path.exists("/var/run/docker.sock"):
        logger.warning("Docker socket not found at /var/run/docker.sock")
        logger.warning("Container management functions will not work")
    
    # Check for NVIDIA runtime
    try:
        info = docker_client.info()
        if "nvidia" not in info.get("Runtimes", {}):
            logger.warning("NVIDIA runtime not found in Docker")
            logger.warning("GPU acceleration may not be available")
    except Exception as e:
        logger.error(f"Error checking Docker runtimes: {str(e)}")
    
    # Create Docker network
    ensure_docker_network()
    
    # Check for HF token
    if not HF_TOKEN:
        logger.warning("No Hugging Face token found. Model loading may fail.")
    else:
        logger.info("Hugging Face token configured")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)