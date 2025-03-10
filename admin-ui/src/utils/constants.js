/**
 * Application constants
 */

// API URL
export const API_URL = import.meta.env.VITE_API_URL || '/api';

// Polling intervals (in milliseconds)
export const POLLING_INTERVALS = {
  SYSTEM_STATS: 5000,
  SYSTEM_METRICS: 3000,
  JOBS: 3000,
  LOGS: 2000,
  DATASETS: 10000
};

// Model configuration options
export const MODEL_CONFIG_OPTIONS = {
  QUANTIZATION: [
    { value: "bitsandbytes", label: "BitsAndBytes 4-bit" },
    { value: "awq", label: "AWQ" },
    { value: "gptq", label: "GPTQ" },
    { value: "gguf", label: "GGUF" },
    { value: null, label: "None (FP16/BF16)" }
  ],
  
  LOAD_FORMAT: [
    { value: "bitsandbytes", label: "BitsAndBytes" },
    { value: "pt", label: "PyTorch" },
    { value: "safetensors", label: "SafeTensors" },
    { value: "auto", label: "Auto-detect" }
  ],
  
  DTYPE: [
    { value: "auto", label: "Auto-detect" },
    { value: "float16", label: "Float16" },
    { value: "bfloat16", label: "BFloat16" },
    { value: "float32", label: "Float32" },
    { value: "half", label: "half" }
  ],
  
  CHAT_TEMPLATE: [
    { value: "llama-3.1", label: "Llama 3.1" },
    { value: "llama-3", label: "Llama 3" },
    { value: "llama-2", label: "Llama 2" },
    { value: "mistral", label: "Mistral" },
    { value: "chatml", label: "ChatML" },
    { value: "zephyr", label: "Zephyr" }
  ]
};

// Job status colors
export const STATUS_COLORS = {
  // For badges and status indicators
  completed: { bg: "bg-green-900", text: "text-green-300" },
  failed: { bg: "bg-red-900", text: "text-red-300" },
  stopped: { bg: "bg-gray-700", text: "text-gray-300" },
  running: { bg: "bg-yellow-900", text: "text-yellow-300" },
  ready: { bg: "bg-green-900", text: "text-green-300" },
  initializing: { bg: "bg-yellow-900", text: "text-yellow-300" },
  starting: { bg: "bg-yellow-900", text: "text-yellow-300" },
  default: { bg: "bg-gray-700", text: "text-gray-300" }
};

// Tab names
export const TABS = {
  OVERVIEW: 'overview',
  MODELS: 'models',
  ADAPTERS: 'adapters',
  DATASETS: 'datasets',
  TRAINING: 'training',
  SERVING: 'serving',
  LOGS: 'logs'
};
