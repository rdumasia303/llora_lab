/**
 * Validation utility functions
 */

// Check if a string is a valid model name
export const isValidModelName = (name) => {
    if (!name) return false;
    
    // Only allow alphanumeric characters, dashes and underscores
    return /^[a-zA-Z0-9-_]+$/.test(name);
  };
  
  // Check if a string is a valid HuggingFace model ID
  export const isValidHfModelId = (id) => {
    if (!id) return false;
    
    // Allow org/model format with alphanumeric, dashes, underscores, and slashes
    return /^[a-zA-Z0-9-_]+\/[a-zA-Z0-9-_]+$/.test(id) || 
           /^[a-zA-Z0-9-_]+$/.test(id); // Allow just model name too
  };
  
  // Check if a number is within a valid range
  export const isNumberInRange = (num, min, max) => {
    if (num === null || num === undefined) return true; // Allow null/undefined as they're optional
    const parsed = parseFloat(num);
    return !isNaN(parsed) && parsed >= min && parsed <= max;
  };
  
  // Check if a string is a valid adapter name
  export const isValidAdapterName = (name) => {
    if (!name) return false;
    
    // Only allow alphanumeric characters, dashes and underscores
    return /^[a-zA-Z0-9-_]+$/.test(name);
  };
  
  // Check if a dataset name is valid
  export const isValidDatasetFileName = (name) => {
    if (!name) return false;
    
    // Must end with .jsonl
    return name.endsWith('.jsonl') && /^[a-zA-Z0-9-_\.]+$/.test(name);
  };
  
  // Validate model form data
  export const validateModelForm = (modelData) => {
    const errors = {};
    
    if (!isValidModelName(modelData.name)) {
      errors.name = "Model name must contain only letters, numbers, dashes, and underscores";
    }
    
    if (!isValidHfModelId(modelData.model_id)) {
      errors.model_id = "Invalid model ID format. Should be 'organization/model-name'";
    }
    
    if (modelData.gpu_memory_utilization && 
        !isNumberInRange(modelData.gpu_memory_utilization, 0, 1)) {
      errors.gpu_memory_utilization = "GPU memory utilization must be between 0 and 1";
    }
    
    return {
      isValid: Object.keys(errors).length === 0,
      errors
    };
  };
  
  // Validate adapter form data
  export const validateAdapterForm = (adapterData) => {
    const errors = {};
    
    if (!isValidAdapterName(adapterData.name)) {
      errors.name = "Adapter name must contain only letters, numbers, dashes, and underscores";
    }
    
    if (!adapterData.base_model) {
      errors.base_model = "Base model is required";
    }
    
    if (!adapterData.dataset) {
      errors.dataset = "Dataset is required";
    }
    
    return {
      isValid: Object.keys(errors).length === 0,
      errors
    };
  };
  