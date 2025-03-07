import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { ChevronRight, Settings, Play, Pause, Monitor, Book, Database, GitBranch, RefreshCw, 
         Terminal, Upload, CheckCircle, Coffee, AlertTriangle, Trash, Edit, Plus, 
         Server, Download, Save, X, Check, HardDrive, Info, ExternalLink, Layers} from 'lucide-react';

// Get API URL from environment or use a default for development
// This will be replaced at build time with the actual API URL
const API_URL = import.meta.env.VITE_API_URL || '/api';

// Utility function to format file sizes
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Formats a timestamp to a readable date
const formatTimestamp = (timestamp) => {
  if (!timestamp) return '';
  try {
    return new Date(timestamp).toLocaleString();
  } catch (e) {
    return timestamp;
  }
};

// Dashboard component that manages all the tabs and state
const AdminDashboard = () => {
  // Reference for intervals to prevent memory leaks
  const refreshIntervalsRef = useRef({});

  // UI state
  const [activeTab, setActiveTab] = useState('overview');
  const [modelsTab, setModelsTab] = useState('list');
  const [adaptersTab, setAdaptersTab] = useState('list');
  const [datasetsTab, setDatasetsTab] = useState('list');
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [showAdapterDialog, setShowAdapterDialog] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [confirmAction, setConfirmAction] = useState(null);
  const [confirmMessage, setConfirmMessage] = useState('');
  
  // Error and loading states
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState({
    models: false,
    adapters: false,
    datasets: false,
    jobs: false,
    system: false,
    logs: false
  });

  // Data states
  const [models, setModels] = useState([]);
  const [adapters, setAdapters] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [servingJobs, setServingJobs] = useState([]);
  const [systemStats, setSystemStats] = useState({
    gpu: { utilized: "0", temperature: "N/A", memory: "0/0 GB" },
    containers: [],
    disk_usage: {}
  });
  const [systemMetrics, setSystemMetrics] = useState({
    timestamps: [],
    gpu_utilization: [],
    memory_usage: []
  });
  const [systemLoading, setSystemLoading] = useState(false);
  
  // Form options
  const [quantizationOptions, setQuantizationOptions] = useState([
    { value: "bitsandbytes", label: "BitsAndBytes 4-bit" },
    { value: "awq", label: "AWQ" },
    { value: "gptq", label: "GPTQ" },
    { value: "gguf", label: "GGUF" },
    { value: null, label: "None (FP16/BF16)" }
  ]);
  
  const [loadFormatOptions, setLoadFormatOptions] = useState([
    { value: "bitsandbytes", label: "BitsAndBytes" },
    { value: "pt", label: "PyTorch" },
    { value: "safetensors", label: "SafeTensors" },
    { value: "auto", label: "Auto-detect" }
  ]);
  
  const [dtypeOptions, setDtypeOptions] = useState([
    { value: "auto", label: "Auto-detect" },
    { value: "float16", label: "Float16" },
    { value: "bfloat16", label: "BFloat16" },
    { value: "float32", label: "Float32" },
    { value: "half", label: "half" }
  ]);
  
  const [chatTemplateOptions, setChatTemplateOptions] = useState([
    { value: "llama-3.1", label: "Llama 3.1" },
    { value: "llama-3", label: "Llama 3" },
    { value: "llama-2", label: "Llama 2" },
    { value: "mistral", label: "Mistral" },
    { value: "chatml", label: "ChatML" },
    { value: "zephyr", label: "Zephyr" }
  ]);
  
  // Test states
  const [selectedAdapter, setSelectedAdapter] = useState("");
  const [testPrompt, setTestPrompt] = useState("");
  const [testResult, setTestResult] = useState("");
  const [isTestLoading, setIsTestLoading] = useState(false);
  const [testParams, setTestParams] = useState({
    temperature: 0.7,
    top_p: 0.9,
    max_tokens: 256
  });
  
  // New model form state
  const [newModel, setNewModel] = useState({
    name: "",
    description: "",
    model_id: "",
    quantization: "",
    load_format: "",
    dtype: "",
    max_model_len: null,
    max_num_seqs: null,
    gpu_memory_utilization: null,
    tensor_parallel_size: null,
    enforce_eager: false,
    trust_remote_code: false,
    chat_template: "",
    response_role: "",
    additional_params: {}
  });
  
  // New adapter form state
  const [newAdapter, setNewAdapter] = useState({
    name: "",
    description: "",
    base_model: "",
    dataset: "",
    lora_rank: null,
    lora_alpha: null,
    lora_dropout: null,
    steps: null,
    batch_size: null,
    gradient_accumulation: null,
    learning_rate: null,
    max_seq_length: null,
    chat_template: "",
    use_nested_quant: false
  });
  
  const [uploadFile, setUploadFile] = useState(null);
  
  // Serving state
  const [servingConfig, setServingConfig] = useState({
    model: "",
    adapter: ""
  });
  
  // Logs state
  const [selectedJobLogs, setSelectedJobLogs] = useState(null);
  const [logContent, setLogContent] = useState("");

  // API request helper with error handling
  const fetchAPI = useCallback(async (endpoint, options = {}) => {
    try {
      const response = await fetch(`${API_URL}${endpoint}`, options);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error (${response.status}): ${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      setError(`Error: ${error.message}`);
      throw error;
    }
  }, []);

  // Data fetching functions
  const fetchModels = useCallback(async () => {
    setLoading(prev => ({ ...prev, models: true }));
    try {
      const data = await fetchAPI('/configs/models');
      setModels(data);
    } catch (error) {
      console.error("Error fetching models:", error);
    } finally {
      setLoading(prev => ({ ...prev, models: false }));
    }
  }, [fetchAPI]);
  
  const fetchAdapters = useCallback(async () => {
    setLoading(prev => ({ ...prev, adapters: true }));
    try {
      const data = await fetchAPI('/adapters');
      setAdapters(data);
    } catch (error) {
      console.error("Error fetching adapters:", error);
    } finally {
      setLoading(prev => ({ ...prev, adapters: false }));
    }
  }, [fetchAPI]);
  
  const fetchDatasets = useCallback(async () => {
    setLoading(prev => ({ ...prev, datasets: true }));
    try {
      const data = await fetchAPI('/datasets');
      setDatasets(data);
    } catch (error) {
      console.error("Error fetching datasets:", error);
    } finally {
      setLoading(prev => ({ ...prev, datasets: false }));
    }
  }, [fetchAPI]);
  
  const fetchJobs = useCallback(async () => {
    setLoading(prev => ({ ...prev, jobs: true }));
    try {
      // Fetch both training and serving jobs
      const trainingJobs = await fetchAPI('/training/jobs');
      const servingJobs = await fetchAPI('/serving/jobs');
      
      setTrainingJobs(trainingJobs);
      setServingJobs(servingJobs);
    } catch (error) {
      console.error("Error fetching jobs:", error);
    } finally {
      setLoading(prev => ({ ...prev, jobs: false }));
    }
  }, [fetchAPI]);
  
  // const fetchSystemStats = useCallback(async () => {
  //   setLoading(prev => ({ ...prev, system: true }));
  //   try {
  //     const data = await fetchAPI('/system/stats');
  //     setSystemStats(data);
  //   } catch (error) {
  //     console.error("Error fetching system stats:", error);
  //   } finally {
  //     setLoading(prev => ({ ...prev, system: false }));
  //   }
  // }, [fetchAPI]);


  const fetchSystemStats = useCallback(async () => {
    setSystemLoading(true);
    try {
      const data = await fetchAPI('/system/stats');
      setSystemStats(data);
    } catch (error) {
      console.error("Error fetching system stats:", error);
    } finally {
      setSystemLoading(false);
    }
  }, [fetchAPI]);

  const fetchSystemMetrics = useCallback(async () => {
    try {
      const data = await fetchAPI('/system/metrics');
      
      // Update metrics history
      setSystemMetrics(prev => {
        // Add new metrics data while maintaining limited history
        const timestamps = [...prev.timestamps, data.timestamp].slice(-30);
        const gpuUtil = [...prev.gpu_utilization, parseFloat(data.gpu_utilization || 0)].slice(-30);
        const memUsage = [...prev.memory_usage, data.memory_usage?.percent_used || 0].slice(-30);
        
        return {
          timestamps,
          gpu_utilization: gpuUtil,
          memory_usage: memUsage
        };
      });
    } catch (error) {
      console.error("Error fetching system metrics:", error);
    }
  }, [fetchAPI]);
  
  const fetchLogs = useCallback(async (jobId, jobType) => {
    if (!jobId) return;
    
    setLoading(prev => ({ ...prev, logs: true }));

    try {
      const endpoint = jobType === 'training' 
        ? `/training/logs/${jobId}`
        : `/serving/logs/${jobId}`;
        
      const data = await fetchAPI(endpoint);
      setLogContent(data.logs || "No logs available");
    } catch (error) {
      console.error("Error fetching logs:", error);
      setLogContent("Error fetching logs: " + error.message);
    } finally {
      setLoading(prev => ({ ...prev, logs: false }));
    }
  }, [fetchAPI]);

  // Clear all refresh intervals to prevent memory leaks
  const clearAllIntervals = useCallback(() => {
    Object.values(refreshIntervalsRef.current).forEach(interval => {
      if (interval) clearInterval(interval);
    });
    refreshIntervalsRef.current = {};
  }, []);

  // Set up interval for a specific task
  const setupInterval = useCallback((key, func, delay) => {
    // Clear existing interval if any
    if (refreshIntervalsRef.current[key]) {
      clearInterval(refreshIntervalsRef.current[key]);
    }
    
    // Set up new interval
    refreshIntervalsRef.current[key] = setInterval(func, delay);
    
    // Return cleanup function
    return () => {
      if (refreshIntervalsRef.current[key]) {
        clearInterval(refreshIntervalsRef.current[key]);
        delete refreshIntervalsRef.current[key];
      }
    };
  }, []);

  // Load data based on active tab - reduced unnecessary fetches
  useEffect(() => {
    // Clean up existing intervals
    clearAllIntervals();
    
    // Initial data loads based on current tab
    const loadInitialData = async () => {
      try {
        // Common data needed across tabs
        if (['overview', 'models', 'adapters', 'datasets', 'training', 'serving'].includes(activeTab)) {
          await Promise.all([
            activeTab === 'overview' || activeTab === 'models' ? fetchModels() : Promise.resolve(),
            activeTab === 'overview' || activeTab === 'adapters' ? fetchAdapters() : Promise.resolve(),
            activeTab === 'overview' || activeTab === 'datasets' ? fetchDatasets() : Promise.resolve(),
            ['overview', 'training', 'serving'].includes(activeTab) ? fetchJobs() : Promise.resolve(),
            activeTab === 'overview' ? fetchSystemStats() : Promise.resolve(),
            activeTab === 'overview' ? fetchSystemMetrics() : Promise.resolve()
          ]);
        }
        
        // For logs tab
        if (activeTab === 'logs' && selectedJobLogs) {
          await fetchLogs(selectedJobLogs.id, selectedJobLogs.type);
        }
      } catch (error) {
        console.error("Error loading initial data:", error);
      }
    };

    loadInitialData();
    
    // Setup appropriate intervals based on active tab
    if (activeTab === 'overview') {
      setupInterval('jobs', fetchJobs, 5000);
      setupInterval('systemStats', fetchSystemStats, 5000);
      setupInterval('systemMetrics', fetchSystemMetrics, 3000);
    } else if (activeTab === 'training' || activeTab === 'serving') {
      setupInterval('jobs', fetchJobs, 3000);
    } else if (activeTab === 'logs' && selectedJobLogs) {
      setupInterval('logs', () => fetchLogs(selectedJobLogs.id, selectedJobLogs.type), 5000);
    }
    
    // Clean up all intervals on component unmount
    return clearAllIntervals;
  }, [activeTab, selectedJobLogs, fetchModels, fetchAdapters, fetchDatasets, fetchJobs, 
      fetchSystemStats, fetchSystemMetrics, fetchLogs, clearAllIntervals, setupInterval]);

  // API Actions
  const createModel = async () => {
    try {
      // Remove any null/undefined values for clean API request
      const modelData = Object.fromEntries(
        Object.entries(newModel).filter(([_, v]) => v !== null && v !== undefined && v !== "")
      );
      
      await fetchAPI('/configs/models', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(modelData)
      });
      
      fetchModels();
      setShowModelDialog(false);
      setNewModel({
        name: "",
        description: "",
        model_id: "",
        quantization: "",
        load_format: "",
        dtype: "",
        max_model_len: null,
        max_num_seqs: null,
        gpu_memory_utilization: null,
        tensor_parallel_size: null,
        enforce_eager: false,
        trust_remote_code: false,
        chat_template: "",
        response_role: "",
        additional_params: {}
      });
    } catch (error) {
      console.error("Error creating model:", error);
    }
  };
  
  const deleteModel = async (name) => {
    try {
      await fetchAPI(`/configs/models/${name}`, {
        method: 'DELETE'
      });
      
      fetchModels();
    } catch (error) {
      console.error("Error deleting model:", error);
    }
  };
  
  const uploadDataset = async () => {
    if (!uploadFile) return;
    
    try {
      const formData = new FormData();
      formData.append('file', uploadFile);
      
      await fetchAPI('/datasets/upload', {
        method: 'POST',
        body: formData
      });
      
      fetchDatasets();
      setUploadFile(null);
      document.getElementById('datasetUpload').value = '';
      setDatasetsTab('list');
    } catch (error) {
      console.error("Error uploading dataset:", error);
    }
  };
  
  const deleteDataset = async (name) => {
    try {
      await fetchAPI(`/datasets/${name}`, {
        method: 'DELETE'
      });
      
      fetchDatasets();
    } catch (error) {
      console.error("Error deleting dataset:", error);
    }
  };
  
  const testModel = async () => {
    if (!testPrompt) return;
    
    setIsTestLoading(true);
    setTestResult("");
    
    try {
      const params = new URLSearchParams({
        prompt: testPrompt,
        temperature: testParams.temperature,
        top_p: testParams.top_p,
        max_tokens: testParams.max_tokens
      });
      
      const data = await fetchAPI(`/test/model?${params.toString()}`, {
        method: 'POST'
      });
      
      setTestResult(data.response);
    } catch (error) {
      console.error("Error testing model:", error);
      setTestResult(`Error: ${error.message}`);
    } finally {
      setIsTestLoading(false);
    }
  };
  
  const testAdapter = async () => {
    if (!selectedAdapter || !testPrompt) return;
    
    setIsTestLoading(true);
    setTestResult("");
    
    try {
      const params = new URLSearchParams({
        adapter_name: selectedAdapter,
        prompt: testPrompt,
        temperature: testParams.temperature,
        top_p: testParams.top_p,
        max_tokens: testParams.max_tokens
      });
      
      const data = await fetchAPI(`/test/adapter?${params.toString()}`, {
        method: 'POST'
      });
      
      setTestResult(data.response);
    } catch (error) {
      console.error("Error testing adapter:", error);
      setTestResult(`Error: ${error.message}`);
    } finally {
      setIsTestLoading(false);
    }
  };
  
  const createAdapter = async () => {
    try {
      // Remove any null/undefined values for clean API request
      const adapterData = Object.fromEntries(
        Object.entries(newAdapter).filter(([_, v]) => v !== null && v !== undefined && v !== "")
      );
      
      await fetchAPI('/configs/adapters', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(adapterData)
      });
      
      fetchAdapters();
      setShowAdapterDialog(false);
    } catch (error) {
      console.error("Error creating adapter config:", error);
    }
  };
  
  const startTraining = async () => {
    try {
      // First create adapter config if it doesn't exist
      await createAdapter();
      
      // Then start training
      await fetchAPI(`/training/start?adapter_name=${newAdapter.name}`, {
        method: 'POST'
      });
      
      fetchJobs();
      setShowAdapterDialog(false);
      setAdaptersTab('list');
    } catch (error) {
      console.error("Error starting training:", error);
    }
  };
  
  const stopTrainingJob = async (jobId) => {
    try {
      await fetchAPI(`/training/jobs/${jobId}`, {
        method: 'DELETE'
      });
      
      fetchJobs();
    } catch (error) {
      console.error("Error stopping training:", error);
    }
  };
  
  const startServing = async () => {
    if (!servingConfig.model) return;
  
    setLoading(prev => ({ ...prev, jobs: true }));

    try {
      const url = `/serving/start?model_name=${servingConfig.model}${servingConfig.adapter ? `&adapter=${servingConfig.adapter}` : ''}`;
      await fetchAPI(url, {
        method: 'POST'
      });

      setLogContent(""); 
      
      fetchJobs();
    } catch (error) {
      console.error("Error starting serving:", error);
    }
  };
  
  const stopServingJob = async (jobId) => {
    try {
      await fetchAPI(`/serving/jobs/${jobId}`, {
        method: 'DELETE'
      });
      
      fetchJobs();
    } catch (error) {
      console.error("Error stopping serving:", error);
    }
  };
  
  const deleteAdapter = async (name) => {
    try {
      await fetchAPI(`/adapters/${name}`, {
        method: 'DELETE'
      });
      
      fetchAdapters();
    } catch (error) {
      console.error("Error deleting adapter:", error);
    }
  };

  // Get active training job if any
  const activeTrainingJob = trainingJobs.find(job => 
    job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped'
  );
  
  // Get active serving job if any
  const activeServingJob = servingJobs.find(job => 
    job.status !== 'stopped' && job.status !== 'failed'
  );
  
  // Create training data for chart
  const trainingChartData = activeTrainingJob && activeTrainingJob.step > 0 ? 
    Array.from({ length: activeTrainingJob.step }, (_, i) => ({
      step: i + 1,
      loss: activeTrainingJob.loss ? (activeTrainingJob.loss - (0.01 * (activeTrainingJob.step - (i + 1)))) : null,
      learningRate: activeTrainingJob.learning_rate
    })) : [];

  // Create system metrics chart data
  const systemMetricsData = systemMetrics.timestamps.map((timestamp, index) => ({
    timestamp,
    gpu: systemMetrics.gpu_utilization[index] || 0,
    memory: systemMetrics.memory_usage[index] || 0
  }));
  
  const handleConfirmAction = () => {
    if (confirmAction) {
      confirmAction();
    }
    setShowConfirmDialog(false);
    setConfirmAction(null);
    setConfirmMessage('');
  };
  
  // Handle form input for additional parameters in model config
  const handleAdditionalParamChange = (key, value) => {
    setNewModel(prev => ({
      ...prev,
      additional_params: {
        ...prev.additional_params,
        [key]: value
      }
    }));
  };
  
  const addAdditionalParam = () => {
    const key = prompt("Enter parameter name (kebab-case):");
    if (!key) return;
    
    const value = prompt("Enter parameter value:");
    if (value === null) return;
    
    handleAdditionalParamChange(key, value);
  };
  
  const removeAdditionalParam = (key) => {
    setNewModel(prev => {
      const newParams = { ...prev.additional_params };
      delete newParams[key];
      return {
        ...prev,
        additional_params: newParams
      };
    });
  };
  
  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-200">
      {/* Header */}
      <header className="bg-gray-800 p-4 border-b border-gray-700 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Coffee className="h-8 w-8 text-purple-400" />
          <h1 className="text-xl font-bold">Llora Lab</h1>
        </div>
        <div className="flex items-center space-x-4">
          <a 
            href="https://github.com/your-repo/llora-lab/docs" 
            target="_blank" 
            rel="noopener noreferrer"
            className="bg-gray-700 px-3 py-1 rounded-md hover:bg-gray-600 transition-colors text-sm flex items-center"
          >
            <Book size={14} className="mr-1" />
            Documentation
          </a>
          <button className="bg-gray-700 px-3 py-1 rounded-md hover:bg-gray-600 transition-colors text-sm flex items-center">
            <Settings size={14} className="mr-1" />
            Settings
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1">
        {/* Sidebar */}
        <aside className="w-48 bg-gray-800 border-r border-gray-700 p-4">
          <nav className="space-y-1">
            <button 
              onClick={() => setActiveTab('overview')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'overview' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <Monitor size={18} />
              <span>Overview</span>
            </button>
            <button 
              onClick={() => setActiveTab('models')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'models' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <Database size={18} />
              <span>Models</span>
            </button>
            <button 
              onClick={() => setActiveTab('adapters')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'adapters' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <GitBranch size={18} />
              <span>Adapters</span>
            </button>
            <button 
              onClick={() => setActiveTab('datasets')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'datasets' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <Layers size={18} />
              <span>Datasets</span>
            </button>
            <button 
              onClick={() => setActiveTab('training')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'training' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <RefreshCw size={18} />
              <span>Training</span>
            </button>
            <button 
              onClick={() => setActiveTab('serving')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'serving' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <Play size={18} />
              <span>Serving</span>
            </button>
            <button 
              onClick={() => setActiveTab('logs')}
              className={`flex items-center space-x-2 w-full text-left p-2 rounded-md ${activeTab === 'logs' ? 'bg-purple-800 text-white' : 'hover:bg-gray-700'}`}>
              <Terminal size={18} />
              <span>Logs</span>
            </button>
          </nav>
        </aside>

        {/* Content area */}
        <main className="flex-1 p-6 overflow-auto">
          {/* Error message display */}
          {error && (
            <div className="bg-red-800 text-white p-3 rounded mb-4 flex items-center">
              <AlertTriangle size={18} className="mr-2" />
              {error}
              <button 
                className="ml-auto text-white" 
                onClick={() => setError(null)}
              >
                <X size={18} />
              </button>
            </div>
          )}
          
          {/* Main content tabs */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold mb-4">System Overview</h2>
              
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold mb-2">System Status</h3>
                  <div className="flex items-center space-x-2 text-green-400">
                    <CheckCircle size={18} />
                    <span>All systems operational</span>
                  </div>
                  <div className="mt-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span>GPU Memory</span>
                      <span>{systemStats.gpu.memory}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2.5">
                      <div 
                        className="bg-purple-600 h-2.5 rounded-full" 
                        style={{
                          width: `${parseFloat(systemStats.gpu.utilized || 0) * 100}%`
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold mb-2">Active Model</h3>
                  {activeServingJob ? (
                    <>
                      <p className="text-xl">{activeServingJob.model_conf}</p>
                      {activeServingJob.adapter && <p className="text-sm text-gray-400">with {activeServingJob.adapter}</p>}
                      <div className="mt-4 flex justify-between items-center">
                        <span className="text-sm text-gray-400">Status: {activeServingJob.status}</span>
                        <button 
                          onClick={() => {
                            setConfirmAction(() => () => stopServingJob(activeServingJob.id));
                            setConfirmMessage(`Stop serving ${activeServingJob.model_conf}?`);
                            setShowConfirmDialog(true);
                          }}
                          className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm"
                        >
                          Stop
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="text-gray-400">No model currently active</p>
                      <div className="mt-4">
                        <button 
                          onClick={() => setActiveTab('serving')}
                          className="bg-purple-700 hover:bg-purple-600 text-white px-3 py-1 rounded text-sm"
                        >
                          Deploy Model
                        </button>
                      </div>
                    </>
                  )}
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold mb-2">Training Status</h3>
                  {activeTrainingJob ? (
                    <>
                      <div className="flex items-center space-x-2 text-yellow-400">
                        <RefreshCw size={18} />
                        <span>Training in progress</span>
                      </div>
                      <div className="mt-4">
                        <p className="text-sm">{activeTrainingJob.adapter_config} adapter</p>
                        <p className="text-xs text-gray-400">
                          Step {activeTrainingJob.step}/{activeTrainingJob.total_steps} • 
                          Loss: {activeTrainingJob.loss?.toFixed(2) || 'N/A'}
                        </p>
                        <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
                          <div 
                            className="bg-yellow-600 h-2.5 rounded-full" 
                            style={{
                              width: `${(activeTrainingJob.step / activeTrainingJob.total_steps) * 100}%`
                            }}
                          ></div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex items-center space-x-2 text-gray-400">
                        <CheckCircle size={18} />
                        <span>No active training jobs</span>
                      </div>
                      <div className="mt-4">
                        <button 
                          onClick={() => {
                            setActiveTab('adapters');
                            setAdaptersTab('create');
                          }}
                          className="bg-purple-700 hover:bg-purple-600 text-white px-3 py-1 rounded text-sm"
                        >
                          Start Training
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold mb-4">System Metrics</h3>
                  <div className="h-64">
                    {systemMetricsData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={systemMetricsData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" />
                          <XAxis 
                            dataKey="timestamp"
                            tickFormatter={(timestamp) => {
                              const date = new Date(timestamp);
                              return `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
                            }}
                            stroke="#9ca3af" 
                          />
                          <YAxis stroke="#9ca3af" />
                          <Tooltip 
                            contentStyle={{backgroundColor: '#374151', borderColor: '#4b5563'}}
                            labelStyle={{color: '#e5e7eb'}}
                            formatter={(value, name) => {
                              return [`${value.toFixed(2)}%`, name === 'gpu' ? 'GPU Usage' : 'Memory Usage'];
                            }}
                            labelFormatter={(timestamp) => {
                              return new Date(timestamp).toLocaleTimeString();
                            }}
                          />
                          <Legend />
                          <Area 
                            type="monotone" 
                            dataKey="gpu" 
                            name="GPU Utilization" 
                            stroke="#8884d8" 
                            fill="#8884d8" 
                            fillOpacity={0.3} 
                          />
                          <Area 
                            type="monotone" 
                            dataKey="memory" 
                            name="Memory Usage" 
                            stroke="#82ca9d" 
                            fill="#82ca9d" 
                            fillOpacity={0.3} 
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-gray-400">
                        <div className="text-center">
                          <RefreshCw size={36} className="mx-auto mb-2 animate-spin" />
                          <p>Collecting metrics data...</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold mb-4">Training Progress</h3>
                  <div className="h-64">
                    {trainingChartData.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trainingChartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" />
                          <XAxis dataKey="step" stroke="#9ca3af" />
                          <YAxis stroke="#9ca3af" />
                          <Tooltip 
                            contentStyle={{backgroundColor: '#374151', borderColor: '#4b5563'}}
                            labelStyle={{color: '#e5e7eb'}}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="loss" stroke="#8884d8" activeDot={{ r: 8 }} />
                          {activeTrainingJob?.learning_rate && (
                            <Line type="monotone" dataKey="learningRate" stroke="#82ca9d" />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-gray-400">
                        <div className="text-center">
                          <AlertTriangle size={36} className="mx-auto mb-2" />
                          <p>No training data available</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 col-span-2">
                  <h3 className="text-lg font-semibold mb-4">Available Resources</h3>
                  
                  <div className="grid grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-700 p-3 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Models</span>
                        <span className="text-lg font-semibold">{models.length}</span>
                      </div>
                      <button 
                        onClick={() => setActiveTab('models')}
                        className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                      >
                        View all <ChevronRight size={14} />
                      </button>
                    </div>
                    
                    <div className="bg-gray-700 p-3 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Adapters</span>
                        <span className="text-lg font-semibold">{adapters.length}</span>
                      </div>
                      <button 
                        onClick={() => setActiveTab('adapters')}
                        className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                      >
                        View all <ChevronRight size={14} />
                      </button>
                    </div>
                    
                    <div className="bg-gray-700 p-3 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Datasets</span>
                        <span className="text-lg font-semibold">{datasets.length}</span>
                      </div>
                      <button 
                        onClick={() => setActiveTab('datasets')}
                        className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                      >
                        View all <ChevronRight size={14} />
                      </button>
                    </div>
                    
                    <div className="bg-gray-700 p-3 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Jobs</span>
                        <span className="text-lg font-semibold">{trainingJobs.length + servingJobs.length}</span>
                      </div>
                      <button 
                        onClick={() => setActiveTab('training')}
                        className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                      >
                        View all <ChevronRight size={14} />
                      </button>
                    </div>
                  </div>
                  
                  {/* Container status */}
                  <h4 className="text-sm font-medium mb-2">Docker Containers</h4>
                  <div className="space-y-2">
                    {systemStats.containers.map(container => (
                      <div key={container.id} className="flex justify-between text-xs bg-gray-750 p-2 rounded">
                        <span className="text-gray-300">{container.name}</span>
                        <span className={`${container.status === 'running' ? 'text-green-400' : 'text-yellow-400'}`}>
                          {container.status}
                        </span>
                      </div>
                    ))}
                    {systemStats.containers.length === 0 && (
                      <div className="text-xs text-gray-400 p-2">
                        No containers running
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold">System Logs</h2>
                <div className="flex space-x-2">
                  <button
                    onClick={() => {
                      if (selectedJobLogs) {
                        fetchLogs(selectedJobLogs.id, selectedJobLogs.type);
                      }
                    }}
                    className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
                    disabled={!selectedJobLogs || loading.logs}
                  >
                    <RefreshCw size={14} className={`mr-1 ${loading.logs ? 'animate-spin' : ''}`} />
                    Refresh Logs
                  </button>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar with job list */}
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 h-[calc(100vh-200px)] overflow-y-auto">
                  <h3 className="text-lg font-semibold mb-4">Jobs</h3>
                  
                  {/* Training jobs */}
                  <div className="mb-6">
                    <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center">
                      <RefreshCw size={14} className="mr-1" /> Training Jobs
                    </h4>
                    
                    <div className="space-y-2">
                      {trainingJobs.length === 0 ? (
                        <p className="text-sm text-gray-500">No training jobs</p>
                      ) : (
                        trainingJobs
                          .slice()
                          .sort((a, b) => new Date(b.start_time) - new Date(a.start_time))
                          .map(job => (
                          <button
                            key={job.id}
                            onClick={() => {
                              setSelectedJobLogs({
                                id: job.id,
                                type: 'training'
                              });
                              fetchLogs(job.id, 'training');
                            }}
                            className={`w-full text-left p-2 rounded-md text-sm ${
                              selectedJobLogs?.id === job.id 
                                ? 'bg-purple-800 text-white' 
                                : 'hover:bg-gray-700'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span>{job.adapter_config}</span>
                              <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                                job.status === 'completed' ? 'bg-green-900 text-green-300' :
                                job.status === 'failed' ? 'bg-red-900 text-red-300' :
                                job.status === 'stopped' ? 'bg-gray-700 text-gray-300' :
                                'bg-yellow-900 text-yellow-300'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              {formatTimestamp(job.start_time)}
                            </div>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                  
                  {/* Serving jobs */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center">
                      <Server size={14} className="mr-1" /> Serving Jobs
                    </h4>
                    
                    <div className="space-y-2">
                      {servingJobs.length === 0 ? (
                        <p className="text-sm text-gray-500">No serving jobs</p>
                      ) : (
                        servingJobs
                          .slice()
                          .sort((a, b) => new Date(b.start_time) - new Date(a.start_time))
                          .map(job => (
                          <button
                            key={job.id}
                            onClick={() => {
                              setSelectedJobLogs({
                                id: job.id,
                                type: 'serving'
                              });
                              fetchLogs(job.id, 'serving');
                            }}
                            className={`w-full text-left p-2 rounded-md text-sm ${
                              selectedJobLogs?.id === job.id 
                                ? 'bg-purple-800 text-white' 
                                : 'hover:bg-gray-700'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span>{job.model_conf}</span>
                              <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                                job.status === 'ready' ? 'bg-green-900 text-green-300' :
                                job.status === 'failed' ? 'bg-red-900 text-red-300' :
                                job.status === 'stopped' ? 'bg-gray-700 text-gray-300' :
                                'bg-yellow-900 text-yellow-300'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                            {job.adapter && (
                              <div className="text-xs text-purple-400">
                                with adapter: {job.adapter}
                              </div>
                            )}
                            <div className="text-xs text-gray-400 mt-1">
                              {formatTimestamp(job.start_time)}
                            </div>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                </div>
                
                {/* Log content */}
                <div className="bg-gray-800 rounded-lg border border-gray-700 h-[calc(100vh-200px)] flex flex-col lg:col-span-3">
                  <div className="bg-gray-750 border-b border-gray-700 p-3 flex justify-between items-center">
                    <div>
                      <h3 className="font-medium">
                        {selectedJobLogs ? (
                          <>
                            Logs: {selectedJobLogs.type === 'training' ? 
                              trainingJobs.find(j => j.id === selectedJobLogs.id)?.adapter_config :
                              servingJobs.find(j => j.id === selectedJobLogs.id)?.model_conf
                            }
                          </>
                        ) : (
                          'Select a job to view logs'
                        )}
                      </h3>
                      {selectedJobLogs && (
                        <div className="text-xs text-gray-400">
                          {selectedJobLogs.type === 'training' ? 
                            `Training Job` :
                            `Serving Job`
                          }
                          {' • '}
                          {selectedJobLogs.type === 'training' ? 
                            trainingJobs.find(j => j.id === selectedJobLogs.id)?.status :
                            servingJobs.find(j => j.id === selectedJobLogs.id)?.status
                          }
                        </div>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <button 
                        className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
                        onClick={() => document.getElementById('log-content').scrollTop = 0}
                        title="Scroll to top"
                      >
                        <ChevronRight className="h-4 w-4 transform rotate-90" />
                      </button>
                      <button 
                        className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
                        onClick={() => {
                          const content = document.getElementById('log-content');
                          content.scrollTop = content.scrollHeight;
                        }}
                        title="Scroll to bottom"
                      >
                        <ChevronRight className="h-4 w-4 transform -rotate-90" />
                      </button>
                      <div className="relative group">
                        <button 
                          className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
                          title="Copy to clipboard"
                          onClick={() => {
                            navigator.clipboard.writeText(logContent);
                            // Show feedback (could be done with a tooltip, but kept simple here)
                            alert('Logs copied to clipboard');
                          }}
                        >
                          <button className="h-4 w-4" />
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  <div 
                    id="log-content"
                    className="flex-1 p-4 bg-gray-900 font-mono text-xs overflow-y-auto"
                  >
                    {loading.logs ? (
                      <div className="h-full flex items-center justify-center">
                        <div className="text-center">
                          <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-purple-500" />
                          <p>Loading logs...</p>
                        </div>
                      </div>
                    ) : !selectedJobLogs ? (
                      <div className="h-full flex items-center justify-center text-gray-500">
                        <div className="text-center">
                          <Terminal size={32} className="mx-auto mb-2" />
                          <p>Select a job from the sidebar to view logs</p>
                        </div>
                      </div>
                    ) : logContent ? (
                      logContent.split('\n').map((line, i) => (
                        <div key={i} className="text-gray-300 whitespace-pre-wrap">{line || '\u00A0'}</div>
                      ))
                    ) : (
                      <div className="h-full flex items-center justify-center text-gray-500">
                        <div className="text-center">
                          <Info size={32} className="mx-auto mb-2" />
                          <p>No logs available for this job</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'training' && (
            <div>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold">Training Jobs</h2>
                <div className="flex space-x-2">
                  <button
                    onClick={fetchJobs}
                    className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
                    disabled={loading.jobs}
                  >
                    <RefreshCw size={14} className={`mr-1 ${loading.jobs ? 'animate-spin' : ''}`} />
                    Refresh Status
                  </button>
                  <button 
                    onClick={() => {
                      setActiveTab('adapters');
                      setAdaptersTab('create');
                    }}
                    className="bg-green-700 hover:bg-green-600 text-white px-3 py-1 rounded text-sm flex items-center"
                  >
                    <Plus size={14} className="mr-1" /> New Training Job
                  </button>
                </div>
              </div>
              
              {loading.jobs && !trainingJobs.length ? (
                <div className="p-6 text-center bg-gray-800 rounded-lg border border-gray-700">
                  <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-purple-500" />
                  <p>Loading training jobs...</p>
                </div>
              ) : trainingJobs.length === 0 ? (
                <div className="p-8 text-center bg-gray-800 rounded-lg border border-gray-700">
                  <Info size={36} className="mx-auto mb-2 text-gray-400" />
                  <p className="text-gray-400 mb-4">No training jobs found</p>
                  <button 
                    onClick={() => {
                      setActiveTab('adapters');
                      setAdaptersTab('create');
                    }}
                    className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded"
                  >
                    Start Your First Training Job
                  </button>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Active Training Job */}
                  {activeTrainingJob && (
                    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                      <div className="flex justify-between items-start mb-6">
                        <div>
                          <h3 className="text-xl font-semibold flex items-center">
                            <RefreshCw className="h-5 w-5 mr-2 text-yellow-400 animate-spin" />
                            Active Training: {activeTrainingJob.adapter_config}
                          </h3>
                          <p className="text-sm text-gray-400 mt-1">Started: {formatTimestamp(activeTrainingJob.start_time)}</p>
                        </div>
                        
                        <button 
                          onClick={() => {
                            setConfirmAction(() => () => stopTrainingJob(activeTrainingJob.id));
                            setConfirmMessage(`Stop training job for ${activeTrainingJob.adapter_config}?`);
                            setShowConfirmDialog(true);
                          }}
                          className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm flex items-center"
                        >
                          <X size={14} className="mr-1" />
                          Stop Training
                        </button>
                      </div>
                      
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                          <div className="mb-4">
                            <div className="flex justify-between text-sm mb-1">
                              <span>Progress</span>
                              <span>{activeTrainingJob.step} / {activeTrainingJob.total_steps} steps ({Math.round((activeTrainingJob.step / activeTrainingJob.total_steps) * 100)}%)</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2.5">
                              <div 
                                className="bg-yellow-600 h-2.5 rounded-full" 
                                style={{
                                  width: `${(activeTrainingJob.step / activeTrainingJob.total_steps) * 100}%`
                                }}
                              ></div>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-750 p-3 rounded-lg">
                              <h4 className="text-xs uppercase text-gray-400 mb-1">Current Loss</h4>
                              <p className="text-xl font-semibold">
                                {activeTrainingJob.loss !== null && activeTrainingJob.loss !== undefined 
                                  ? activeTrainingJob.loss.toFixed(4) 
                                  : 'N/A'}
                              </p>
                            </div>
                            
                            <div className="bg-gray-750 p-3 rounded-lg">
                              <h4 className="text-xs uppercase text-gray-400 mb-1">Learning Rate</h4>
                              <p className="text-xl font-semibold">
                                {activeTrainingJob.learning_rate !== null && activeTrainingJob.learning_rate !== undefined 
                                  ? activeTrainingJob.learning_rate.toExponential(2) 
                                  : 'N/A'}
                              </p>
                            </div>
                          </div>
                          
                          {activeTrainingJob.message && (
                            <div className="mt-4 bg-gray-750 p-3 rounded-lg">
                              <h4 className="text-xs uppercase text-gray-400 mb-1">Status Message</h4>
                              <p className="text-sm">{activeTrainingJob.message}</p>
                            </div>
                          )}
                          
                          <div className="mt-4">
                            <button
                              onClick={() => {
                                setSelectedJobLogs({
                                  id: activeTrainingJob.id,
                                  type: 'training'
                                });
                                setActiveTab('logs');
                              }}
                              className="text-purple-400 hover:text-purple-300 text-sm flex items-center"
                            >
                              <Terminal size={14} className="mr-1" />
                              View Full Logs
                            </button>
                          </div>
                        </div>
                        
                        <div className="h-64">
                          {trainingChartData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingChartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" />
                                <XAxis dataKey="step" stroke="#9ca3af" />
                                <YAxis stroke="#9ca3af" />
                                <Tooltip 
                                  contentStyle={{backgroundColor: '#374151', borderColor: '#4b5563'}}
                                  labelStyle={{color: '#e5e7eb'}}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="loss" stroke="#8884d8" activeDot={{ r: 8 }} />
                              </LineChart>
                            </ResponsiveContainer>
                          ) : (
                            <div className="h-full flex items-center justify-center text-gray-400">
                              <div className="text-center">
                                <AlertTriangle size={36} className="mx-auto mb-2" />
                                <p>No training data available</p>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Past Training Jobs */}
                  <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                    <h3 className="p-4 border-b border-gray-700 text-lg font-semibold">Training History</h3>
                    
                    <table className="min-w-full divide-y divide-gray-700">
                      <thead className="bg-gray-750">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Adapter</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Progress</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Start Time</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-700">
                        {trainingJobs
                          .slice()
                          .sort((a, b) => new Date(b.start_time) - new Date(a.start_time))
                          .map(job => (
                          <tr key={job.id} className={job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped' ? 'bg-yellow-900 bg-opacity-20' : ''}>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm font-medium">{job.adapter_config}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                                job.status === 'completed' ? 'bg-green-900 text-green-300' :
                                job.status === 'failed' ? 'bg-red-900 text-red-300' :
                                job.status === 'stopped' ? 'bg-gray-700 text-gray-300' :
                                'bg-yellow-900 text-yellow-300'
                              }`}>
                                {job.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                              {job.status === 'completed' ? (
                                '100%'
                              ) : (
                                <div>
                                  <div className="text-xs mb-1">{job.step} / {job.total_steps} steps</div>
                                  <div className="w-32 bg-gray-700 rounded-full h-1.5">
                                    <div 
                                      className={`h-1.5 rounded-full ${
                                        job.status === 'failed' || job.status === 'stopped' ? 'bg-red-600' : 'bg-yellow-600'
                                      }`}
                                      style={{
                                        width: `${(job.step / job.total_steps) * 100}%`
                                      }}
                                    ></div>
                                  </div>
                                </div>
                              )}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                              {formatTimestamp(job.start_time)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                              <div className="flex space-x-3">
                                <button
                                  onClick={() => {
                                    setSelectedJobLogs({
                                      id: job.id,
                                      type: 'training'
                                    });
                                    setActiveTab('logs');
                                  }}
                                  className="text-blue-400 hover:text-blue-300"
                                >
                                  Logs
                                </button>
                                
                                {job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped' && (
                                  <button
                                    onClick={() => {
                                      setConfirmAction(() => () => stopTrainingJob(job.id));
                                      setConfirmMessage(`Stop training job for ${job.adapter_config}?`);
                                      setShowConfirmDialog(true);
                                    }}
                                    className="text-red-400 hover:text-red-300"
                                  >
                                    Stop
                                  </button>
                                )}
                                
                                {job.status === 'completed' && (
                                  <button
                                    onClick={() => {
                                      setServingConfig({
                                        model: adapters.find(a => a.name === job.adapter_config)?.base_model || '',
                                        adapter: job.adapter_config
                                      });
                                      setActiveTab('serving');
                                    }}
                                    className="text-green-400 hover:text-green-300"
                                  >
                                    Deploy
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                        
                        {trainingJobs.length === 0 && (
                          <tr>
                            <td colSpan="5" className="px-6 py-8 text-center text-gray-400">
                              No training jobs found
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'serving' && (
            <div>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold">Model Serving</h2>
                <div className="flex space-x-2">
                  <button
                    onClick={fetchJobs}
                    className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
                    disabled={loading.jobs}
                  >
                    <RefreshCw size={14} className={`mr-1 ${loading.jobs ? 'animate-spin' : ''}`} />
                    Refresh Status
                  </button>
                </div>
              </div>
              
              {error && (
                <div className="bg-red-800 text-white p-3 rounded mb-4 flex items-center">
                  <AlertTriangle size={18} className="mr-2" />
                  {error}
                  <button 
                    className="ml-auto text-white" 
                    onClick={() => setError(null)}
                  >
                    <X size={18} />
                  </button>
                </div>
              )}
              
              {loading.jobs && !activeServingJob && servingJobs.length === 0 ? (
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
                  <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-purple-500" />
                  <p>Loading serving status...</p>
                </div>
              ) : activeServingJob ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Active serving details */}
                  <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 lg:col-span-2">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-xl font-semibold mb-1">Active Model: {activeServingJob.model_conf}</h3>
                        {activeServingJob.adapter && (
                          <div className="text-sm text-purple-400 mb-2">
                            with adapter: {activeServingJob.adapter}
                          </div>
                        )}
                        <div className="flex items-center mb-4">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            activeServingJob.status === 'running' || activeServingJob.status === 'ready' 
                              ? 'bg-green-900 text-green-300' 
                              : activeServingJob.status === 'initializing' || activeServingJob.status === 'starting'
                              ? 'bg-yellow-900 text-yellow-300'
                              : 'bg-red-900 text-red-300'
                          }`}>
                            {activeServingJob.status === 'ready' ? (
                              <><CheckCircle size={12} className="mr-1" /> Ready</>
                            ) : activeServingJob.status === 'running' ? (
                              <><RefreshCw size={12} className="mr-1 animate-spin" /> Running</>
                            ) : activeServingJob.status === 'initializing' || activeServingJob.status === 'starting' ? (
                              <><RefreshCw size={12} className="mr-1 animate-spin" /> Starting</>
                            ) : (
                              <><AlertTriangle size={12} className="mr-1" /> {activeServingJob.status}</>
                            )}
                          </span>
                        </div>
                      </div>
                      <button 
                        onClick={() => {
                          setConfirmAction(() => () => stopServingJob(activeServingJob.id));
                          setConfirmMessage(`Stop serving model ${activeServingJob.model_conf}?`);
                          setShowConfirmDialog(true);
                        }}
                        className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm flex items-center"
                      >
                        <Pause size={14} className="mr-1" />
                        Stop Serving
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="bg-gray-750 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-gray-400 mb-2">Status</h4>
                        <p className="text-lg font-medium">
                          {activeServingJob.status === 'ready' ? 'Online' : activeServingJob.status}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          Started: {formatTimestamp(activeServingJob.start_time)}
                        </p>
                      </div>
                      
                      <div className="bg-gray-750 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-gray-400 mb-2">Requests</h4>
                        <p className="text-lg font-medium">{activeServingJob.requests_served}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          Avg. response time: {activeServingJob.avg_response_time.toFixed(2)}ms
                        </p>
                      </div>
                    </div>
                    
                    {activeServingJob.message && (
                      <div className="bg-gray-750 p-3 rounded-lg mb-4 text-sm">
                        <h4 className="text-xs font-medium text-gray-400 mb-1">Message</h4>
                        <p className="text-gray-300">{activeServingJob.message}</p>
                      </div>
                    )}
                    
                    {activeServingJob.status === 'ready' && (
                      <div className="bg-gray-750 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-gray-400 mb-2">API Endpoints</h4>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm">OpenAI API:</span>
                            <code className="bg-gray-800 px-2 py-1 rounded text-xs">http://localhost:8000/v1</code>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm">Web UI:</span>
                            <a 
                              href="http://localhost:3000" 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="flex items-center text-purple-400 hover:text-purple-300 text-xs"
                            >
                              http://localhost:3000 <ExternalLink size={12} className="ml-1" />
                            </a>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Test interface */}
                  <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                    <h3 className="text-lg font-semibold mb-4">Quick Test</h3>
                    
                    <div className="mb-4">
                      <label className="block text-sm font-medium mb-1">Prompt</label>
                      <textarea 
                        className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200 h-32"
                        placeholder="Enter a prompt to test..."
                        value={testPrompt}
                        onChange={(e) => setTestPrompt(e.target.value)}
                      ></textarea>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-3 mb-4">
                      <div>
                        <label className="block text-xs font-medium mb-1">Temperature</label>
                        <input 
                          type="number" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-1 px-2 text-gray-200"
                          min="0"
                          max="2"
                          step="0.1"
                          value={testParams.temperature}
                          onChange={(e) => setTestParams({...testParams, temperature: parseFloat(e.target.value)})}
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium mb-1">Top P</label>
                        <input 
                          type="number" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-1 px-2 text-gray-200"
                          min="0"
                          max="1"
                          step="0.05"
                          value={testParams.top_p}
                          onChange={(e) => setTestParams({...testParams, top_p: parseFloat(e.target.value)})}
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium mb-1">Max Tokens</label>
                        <input 
                          type="number" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-1 px-2 text-gray-200"
                          min="1"
                          step="1"
                          value={testParams.max_tokens}
                          onChange={(e) => setTestParams({...testParams, max_tokens: parseInt(e.target.value)})}
                        />
                      </div>
                    </div>
                    
                    <button
                      onClick={testModel}
                      disabled={isTestLoading || !testPrompt || activeServingJob.status !== 'ready'}
                      className={`w-full ${
                        activeServingJob.status === 'ready' 
                          ? 'bg-purple-700 hover:bg-purple-600' 
                          : 'bg-gray-600'
                      } text-white py-2 rounded flex items-center justify-center ${
                        (isTestLoading || !testPrompt || activeServingJob.status !== 'ready') ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                    >
                      {isTestLoading ? (
                        <>
                          <RefreshCw size={16} className="mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Play size={16} className="mr-2" />
                          Test Model
                        </>
                      )}
                    </button>
                    
                    {testResult && (
                      <div className="mt-4">
                        <label className="block text-sm font-medium mb-1">Response</label>
                        <div className="bg-gray-750 border border-gray-700 rounded-md p-3 text-gray-300 text-sm max-h-64 overflow-y-auto">
                          {testResult.split('\n').map((line, i) => (
                            <div key={i}>{line || <br />}</div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Deployment form */}
                  <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 lg:col-span-2">
                    <h3 className="text-lg font-semibold mb-4">Deploy a Model</h3>
                    
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div>
                        <label className="block text-sm font-medium mb-1">Model</label>
                        <select
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={servingConfig.model}
                          onChange={(e) => setServingConfig({...servingConfig, model: e.target.value})}
                        >
                          <option value="">Select a model</option>
                          {models.map(model => (
                            <option key={model.name} value={model.name}>
                              {model.name} {model.description ? `- ${model.description}` : ''}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Adapter (Optional)</label>
                        <select
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={servingConfig.adapter}
                          onChange={(e) => setServingConfig({...servingConfig, adapter: e.target.value})}
                          disabled={!servingConfig.model}
                        >
                          <option value="">No adapter</option>
                          {adapters
                            .filter(adapter => !servingConfig.model || adapter.base_model === servingConfig.model)
                            .map(adapter => (
                              <option key={adapter.name} value={adapter.name}>{adapter.name}</option>
                            ))}
                        </select>
                        {servingConfig.model && adapters.length > 0 && !adapters.some(a => a.base_model === servingConfig.model) && (
                          <p className="text-xs text-yellow-400 mt-1">
                            No compatible adapters found for this model.
                          </p>
                        )}
                      </div>
                    </div>
                    
                    {servingConfig.model && (
                      <div className="bg-gray-750 p-4 rounded-lg mb-6">
                        <h4 className="text-sm font-medium text-gray-400 mb-2">Selected Model Details</h4>
                        {models.filter(m => m.name === servingConfig.model).map(model => (
                          <div key={model.name} className="space-y-2">
                            <div>
                              <span className="text-sm font-medium">ID:</span> 
                              <span className="text-sm ml-2 text-gray-300">{model.model_id}</span>
                            </div>
                            {model.description && (
                              <div>
                                <span className="text-sm font-medium">Description:</span>
                                <span className="text-sm ml-2 text-gray-300">{model.description}</span>
                              </div>
                            )}
                            <div className="flex flex-wrap gap-1 mt-2">
                              {model.quantization && (
                                <span className="px-2 py-1 inline-flex rounded-full bg-gray-600 text-xs">
                                  quant: {model.quantization}
                                </span>
                              )}
                              {model.max_model_len && (
                                <span className="px-2 py-1 inline-flex rounded-full bg-gray-600 text-xs">
                                  len: {model.max_model_len}
                                </span>
                              )}
                              {model.tensor_parallel_size && (
                                <span className="px-2 py-1 inline-flex rounded-full bg-green-600 text-xs">
                                  tensor parallel: {model.tensor_parallel_size}
                                </span>
                              )}
                              {model.chat_template && (
                                <span className="px-2 py-1 inline-flex rounded-full bg-gray-600 text-xs">
                                  template: {model.chat_template}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    <button
                      onClick={startServing}
                      disabled={!servingConfig.model || loading.jobs}
                      className={`bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded flex items-center ${
                        !servingConfig.model || loading.jobs ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                    >
                      {loading.jobs ? (
                        <>
                          <RefreshCw size={18} className="mr-2 animate-spin" />
                          Starting...
                        </>
                      ) : (
                        <>
                          <Play size={18} className="mr-2" />
                          Start Serving
                        </>
                      )}
                    </button>
                  </div>
                  
                  {/* Resources and info */}
                  <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                    <h3 className="text-lg font-semibold mb-4">Serving Information</h3>
                    
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-400 mb-2">System Resources</h4>
                        <div className="space-y-2">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>GPU Memory</span>
                              <span>{systemStats.gpu.memory}</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2.5">
                              <div 
                                className="bg-purple-600 h-2.5 rounded-full" 
                                style={{
                                  width: `${parseFloat(systemStats.gpu.utilized || 0) * 100}%`
                                }}
                              ></div>
                            </div>
                          </div>
                          
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span>GPU Temperature</span>
                              <span>{systemStats.gpu.temperature}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-400 mb-2">Available Models</h4>
                        <div className="text-sm">{models.length} models configured</div>
                        <button 
                          onClick={() => setActiveTab('models')}
                          className="text-xs text-purple-400 hover:text-purple-300 flex items-center mt-1"
                        >
                          Manage models <ChevronRight size={12} />
                        </button>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium text-gray-400 mb-2">Available Adapters</h4>
                        <div className="text-sm">{adapters.length} adapters available</div>
                        <button 
                          onClick={() => setActiveTab('adapters')}
                          className="text-xs text-purple-400 hover:text-purple-300 flex items-center mt-1"
                        >
                          Manage adapters <ChevronRight size={12} />
                        </button>
                      </div>
                      
                      <div className="bg-purple-900 bg-opacity-40 p-3 rounded-lg">
                        <h4 className="text-sm font-medium mb-2 flex items-center">
                          <Info size={14} className="mr-1" />
                          Usage Information
                        </h4>
                        <p className="text-xs">
                          When a model is deployed, it exposes a compatible OpenAI API endpoint 
                          at <code className="bg-gray-800 px-1 rounded">http://localhost:8000/v1</code> and 
                          a web UI at <code className="bg-gray-800 px-1 rounded">http://localhost:3000</code>.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Logs section */}
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-4">Serving Logs</h3>
                
                {activeServingJob ? (
                  <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                    <div className="bg-gray-750 px-4 py-2 flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Logs for {activeServingJob.model_conf} {activeServingJob.adapter && `(${activeServingJob.adapter})`}
                      </span>
                      <button 
                        onClick={() => fetchLogs(activeServingJob.id, 'serving')}
                        className="text-gray-400 hover:text-white"
                      >
                        <RefreshCw size={14} className={loading.logs ? "animate-spin" : ""} />
                      </button>
                    </div>
                    <div className="p-4 max-h-64 overflow-y-auto bg-gray-900 font-mono text-xs">
                      {loading.logs ? (
                        <div className="text-center py-4">
                          <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-purple-500" />
                          <p>Loading logs...</p>
                        </div>
                      ) : logContent ? (
                        logContent.split('\n').map((line, i) => (
                          <div key={i} className="text-gray-300">{line || <br />}</div>
                        ))
                      ) : (
                        <p className="text-gray-500">No logs available yet. Start a serving job to see logs.</p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
                    <p className="text-gray-400">No active serving job. Start a model to view logs.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Model management tab */}
          {activeTab === 'models' && (
            <div>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold">Models</h2>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setModelsTab('list')}
                    className={`px-3 py-1 rounded text-sm ${modelsTab === 'list' ? 'bg-purple-700 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                  >
                    Model List
                  </button>
                  <button 
                    onClick={() => {
                      setShowModelDialog(true);
                      setNewModel({
                        name: "",
                        description: "",
                        model_id: "",
                        quantization: "",
                        load_format: "",
                        dtype: "",
                        max_model_len: null,
                        max_num_seqs: null,
                        gpu_memory_utilization: null,
                        tensor_parallel_size: null,
                        enforce_eager: false,
                        trust_remote_code: false,
                        chat_template: "",
                        response_role: "",
                        additional_params: {}
                      });
                    }}
                    className="bg-green-700 hover:bg-green-600 text-white px-3 py-1 rounded text-sm flex items-center"
                  >
                    <Plus size={14} className="mr-1" /> Add Model
                  </button>
                </div>
              </div>
              
              {modelsTab === 'list' && (
                <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                  {loading.models ? (
                    <div className="p-6 text-center">
                      <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-purple-500" />
                      <p>Loading models...</p>
                    </div>
                  ) : models.length === 0 ? (
                    <div className="p-6 text-center">
                      <Info size={36} className="mx-auto mb-2 text-gray-400" />
                      <p className="text-gray-400 mb-4">No models configured yet</p>
                      <button 
                        onClick={() => {
                          setShowModelDialog(true);
                          setNewModel({
                            name: "",
                            description: "",
                            model_id: "",
                            quantization: "",
                            load_format: "",
                            dtype: "",
                            max_model_len: null,
                            max_num_seqs: null,
                            gpu_memory_utilization: null,
                            tensor_parallel_size: null,
                            enforce_eager: false,
                            trust_remote_code: false,
                            chat_template: "",
                            response_role: "",
                            additional_params: {}
                          });
                        }}
                        className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded"
                      >
                        Add Your First Model
                      </button>
                    </div>
                  ) : (
                    <table className="min-w-full divide-y divide-gray-700">
                      <thead className="bg-gray-700">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Model</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">ID</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Parameters</th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-700">
                        {models.map(model => (
                          <tr key={model.name} className="hover:bg-gray-750">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm font-medium">{model.name}</div>
                              <div className="text-xs text-gray-400">{model.description}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">{model.model_id}</td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-xs flex flex-wrap gap-1">
                                {model.quantization && (
                                  <span className="px-2 py-1 inline-flex rounded-full bg-gray-600">
                                    quant: {model.quantization}
                                  </span>
                                )}
                                {model.max_model_len && (
                                  <span className="px-2 py-1 inline-flex rounded-full bg-gray-600">
                                    len: {model.max_model_len}
                                  </span>
                                )}
                                {model.tensor_parallel_size > 1 && (
                                  <span className="px-2 py-1 inline-flex rounded-full bg-green-600">
                                    TP: {model.tensor_parallel_size}
                                  </span>
                                )}
                                {model.enforce_eager && (
                                  <span className="px-2 py-1 inline-flex rounded-full bg-gray-600">
                                    eager: true
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                              <button 
                                onClick={() => {
                                  setServingConfig({
                                    model: model.name,
                                    adapter: ""
                                  });
                                  setActiveTab('serving');
                                }}
                                className="text-blue-400 hover:text-blue-300 mr-3"
                              >
                                Deploy
                              </button>
                              <button 
                                onClick={() => {
                                  setConfirmAction(() => () => deleteModel(model.name));
                                  setConfirmMessage(`Delete model configuration "${model.name}"?`);
                                  setShowConfirmDialog(true);
                                }}
                                className="text-red-400 hover:text-red-300"
                              >
                                Delete
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
              
              {/* Model Dialog */}
              {showModelDialog && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                  <div className="bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold">Add Model Configuration</h3>
                      <button 
                        onClick={() => setShowModelDialog(false)}
                        className="text-gray-400 hover:text-gray-200"
                      >
                        <X size={24} />
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">Name</label>
                        <input 
                          type="text" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.name}
                          onChange={(e) => setNewModel({...newModel, name: e.target.value})}
                          placeholder="e.g., llama-3.1-8b"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Model ID (HuggingFace)</label>
                        <input 
                          type="text" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.model_id}
                          onChange={(e) => setNewModel({...newModel, model_id: e.target.value})}
                          placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct"
                        />
                      </div>
                      
                      <div className="col-span-2">
                        <label className="block text-sm font-medium mb-1">Description</label>
                        <input 
                          type="text" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.description}
                          onChange={(e) => setNewModel({...newModel, description: e.target.value})}
                          placeholder="Brief description of the model"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Quantization Method</label>
                        <select 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.quantization}
                          onChange={(e) => setNewModel({...newModel, quantization: e.target.value || null})}
                        >
                          <option value="">Select quantization (optional)</option>
                          {quantizationOptions.map(option => (
                            <option key={option.value || 'none'} value={option.value || ''}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Load Format</label>
                        <select 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.load_format}
                          onChange={(e) => setNewModel({...newModel, load_format: e.target.value || null})}
                        >
                          <option value="">Select load format (optional)</option>
                          {loadFormatOptions.map(option => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Data Type</label>
                        <select 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.dtype}
                          onChange={(e) => setNewModel({...newModel, dtype: e.target.value || null})}
                        >
                          <option value="">Select data type (optional)</option>
                          {dtypeOptions.map(option => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Chat Template</label>
                        <select 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.chat_template}
                          onChange={(e) => setNewModel({...newModel, chat_template: e.target.value || null})}
                        >
                          <option value="">Select chat template (optional)</option>
                          {chatTemplateOptions.map(option => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Max Model Length</label>
                        <input 
                          type="number" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.max_model_len || ''}
                          onChange={(e) => setNewModel({...newModel, max_model_len: e.target.value ? parseInt(e.target.value) : null})}
                          placeholder="Optional"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">GPU Memory Utilization</label>
                        <input 
                          type="number" 
                          step="0.01" 
                          min="0.1" 
                          max="1.0"
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.gpu_memory_utilization || ''}
                          onChange={(e) => setNewModel({...newModel, gpu_memory_utilization: e.target.value ? parseFloat(e.target.value) : null})}
                          placeholder="Optional (0.1-1.0)"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Tensor Parallel Size</label>
                        <input 
                          type="number" 
                          min="1"
                          step="1"
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.tensor_parallel_size || ''}
                          onChange={(e) => setNewModel({...newModel, tensor_parallel_size: e.target.value ? parseInt(e.target.value) : null})}
                          placeholder="Optional (default: 1)"
                        />
                        <div className="text-xs text-gray-400 mt-1">
                          Use for multi-GPU inference (# GPUs to use)
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Response Role</label>
                        <input 
                          type="text" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.response_role || ''}
                          onChange={(e) => setNewModel({...newModel, response_role: e.target.value || null})}
                          placeholder="Optional (e.g., assistant)"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">Max Number of Sequences</label>
                        <input 
                          type="number" 
                          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
                          value={newModel.max_num_seqs || ''}
                          onChange={(e) => setNewModel({...newModel, max_num_seqs: e.target.value ? parseInt(e.target.value) : null})}
                          placeholder="Optional"
                        />
                      </div>
                      
                      <div className="flex items-center col-span-2">
                        <div className="flex items-center mr-6">
                          <input 
                            type="checkbox" 
                            id="enforce_eager" 
                            className="mr-2"
                            checked={newModel.enforce_eager || false}
                            onChange={(e) => setNewModel({...newModel, enforce_eager: e.target.checked})}
                          />
                          <label htmlFor="enforce_eager" className="text-sm font-medium">Enforce Eager Mode</label>
                        </div>
                        
                        <div className="flex items-center">
                          <input 
                            type="checkbox" 
                            id="trust_remote_code" 
                            className="mr-2"
                            checked={newModel.trust_remote_code || false}
                            onChange={(e) => setNewModel({...newModel, trust_remote_code: e.target.checked})}
                          />
                          <label htmlFor="trust_remote_code" className="text-sm font-medium">Trust Remote Code</label>
                        </div>
                      </div>
                      
                      {/* Additional Parameters */}
                      <div className="col-span-2 mt-4">
                        <div className="flex justify-between items-center mb-2">
                          <label className="block text-sm font-medium">Additional Parameters</label>
                          <button 
                            type="button"
                            onClick={addAdditionalParam}
                            className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                          >
                            <Plus size={14} className="mr-1" /> Add Parameter
                          </button>
                        </div>
                        
                        {Object.keys(newModel.additional_params).length === 0 ? (
                          <div className="bg-gray-700 border border-gray-600 rounded-md p-4 text-sm text-gray-400">
                            No additional parameters. Click "Add Parameter" to add custom vLLM parameters.
                          </div>
                        ) : (
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(newModel.additional_params).map(([key, value]) => (
                              <div key={key} className="bg-gray-700 p-2 rounded-md flex items-center">
                                <div className="flex-1">
                                  <span className="text-xs font-medium text-gray-300">{key}</span>
                                  <span className="text-xs text-gray-400 ml-2">{value.toString()}</span>
                                </div>
                                <button 
                                  onClick={() => removeAdditionalParam(key)}
                                  className="text-red-400 hover:text-red-300"
                                >
                                  <X size={14} />
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="mt-6 flex justify-end space-x-2">
                      <button 
                        onClick={() => setShowModelDialog(false)}
                        className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
                      >
                        Cancel
                      </button>
                      <button 
                        onClick={createModel}
                        className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded-md"
                        disabled={!newModel.name || !newModel.model_id}
                      >
                        Save Model
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Confirm dialog */}
          {showConfirmDialog && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full">
                <div className="flex flex-col items-center mb-4">
                  <AlertTriangle size={48} className="text-yellow-500 mb-2" />
                  <h3 className="text-lg font-semibold text-center">{confirmMessage || "Are you sure?"}</h3>
                </div>
                
                <div className="flex justify-center space-x-4">
                  <button 
                    onClick={() => setShowConfirmDialog(false)}
                    className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
                  >
                    Cancel
                  </button>
                  <button 
                    onClick={handleConfirmAction}
                    className="bg-red-700 hover:bg-red-600 text-white px-4 py-2 rounded-md"
                  >
                    Confirm
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default AdminDashboard;