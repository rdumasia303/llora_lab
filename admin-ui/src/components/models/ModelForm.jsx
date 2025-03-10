import React, { useState, useEffect } from 'react';
import { X, Plus, Save } from 'lucide-react';
import { MODEL_CONFIG_OPTIONS } from '../../utils/constants';
import { validateModelForm } from '../../utils/validators';

/**
 * Form for adding/editing model configurations
 */
const ModelForm = ({ 
  model, 
  isEditMode = false, 
  onSave, 
  onCancel 
}) => {
  const [formData, setFormData] = useState({
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
  
  const [errors, setErrors] = useState({});
  
  // Initialize form with model data if editing
  useEffect(() => {
    if (model && isEditMode) {
      setFormData(prevData => ({
        ...prevData,
        ...model
      }));
    }
  }, [model, isEditMode]);
  
  // Handle form field changes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
    
    // Clear error for this field if it exists
    if (errors[name]) {
      setErrors(prev => {
        const newErrors = {...prev};
        delete newErrors[name];
        return newErrors;
      });
    }
  };
  
  // Handle number inputs
  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    let parsedValue = value === '' ? null : 
                   name === 'gpu_memory_utilization' ? parseFloat(value) : 
                   parseInt(value);
    
    setFormData(prev => ({
      ...prev,
      [name]: parsedValue
    }));
  };
  
  // Handle additional parameters
  const handleAdditionalParamChange = (key, value) => {
    setFormData(prev => ({
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
    setFormData(prev => {
      const newParams = { ...prev.additional_params };
      delete newParams[key];
      
      return {
        ...prev,
        additional_params: newParams
      };
    });
  };
  
  // Form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate form
    const validation = validateModelForm(formData);
    if (!validation.isValid) {
      setErrors(validation.errors);
      return;
    }
    
    // Remove null/undefined values
    const cleanedData = Object.fromEntries(
      Object.entries(formData).filter(([_, v]) => v !== null && v !== undefined && v !== "")
    );
    
    onSave(cleanedData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">
            Name {isEditMode && <span className="text-gray-400">(read-only)</span>}
          </label>
          <input 
            type="text" 
            className={`w-full bg-gray-700 border ${errors.name ? 'border-red-500' : 'border-gray-600'} rounded-md py-2 px-3 text-gray-200`}
            value={formData.name}
            name="name"
            onChange={handleChange}
            placeholder="e.g., llama-3.1-8b"
            readOnly={isEditMode}
            disabled={isEditMode}
          />
          {errors.name && <p className="text-xs text-red-400 mt-1">{errors.name}</p>}
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Model ID (HuggingFace)</label>
          <input 
            type="text" 
            className={`w-full bg-gray-700 border ${errors.model_id ? 'border-red-500' : 'border-gray-600'} rounded-md py-2 px-3 text-gray-200`}
            value={formData.model_id}
            name="model_id"
            onChange={handleChange}
            placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct"
          />
          {errors.model_id && <p className="text-xs text-red-400 mt-1">{errors.model_id}</p>}
        </div>
        
        <div className="col-span-2">
          <label className="block text-sm font-medium mb-1">Description</label>
          <input 
            type="text" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.description || ''}
            name="description"
            onChange={handleChange}
            placeholder="Brief description of the model"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Quantization Method</label>
          <select 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.quantization || ''}
            name="quantization"
            onChange={handleChange}
          >
            <option value="">Select quantization (optional)</option>
            {MODEL_CONFIG_OPTIONS.QUANTIZATION.map(option => (
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
            value={formData.load_format || ''}
            name="load_format"
            onChange={handleChange}
          >
            <option value="">Select load format (optional)</option>
            {MODEL_CONFIG_OPTIONS.LOAD_FORMAT.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Data Type</label>
          <select 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.dtype || ''}
            name="dtype"
            onChange={handleChange}
          >
            <option value="">Select data type (optional)</option>
            {MODEL_CONFIG_OPTIONS.DTYPE.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Chat Template</label>
          <select 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.chat_template || ''}
            name="chat_template"
            onChange={handleChange}
          >
            <option value="">Select chat template (optional)</option>
            {MODEL_CONFIG_OPTIONS.CHAT_TEMPLATE.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Max Model Length</label>
          <input 
            type="number" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.max_model_len || ''}
            name="max_model_len"
            onChange={handleNumberChange}
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
            className={`w-full bg-gray-700 border ${errors.gpu_memory_utilization ? 'border-red-500' : 'border-gray-600'} rounded-md py-2 px-3 text-gray-200`}
            value={formData.gpu_memory_utilization || ''}
            name="gpu_memory_utilization"
            onChange={handleNumberChange}
            placeholder="Optional (0.1-1.0)"
          />
          {errors.gpu_memory_utilization && (
            <p className="text-xs text-red-400 mt-1">{errors.gpu_memory_utilization}</p>
          )}
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Tensor Parallel Size</label>
          <input 
            type="number" 
            min="1"
            step="1"
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.tensor_parallel_size || ''}
            name="tensor_parallel_size"
            onChange={handleNumberChange}
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
            value={formData.response_role || ''}
            name="response_role"
            onChange={handleChange}
            placeholder="Optional (e.g., assistant)"
          />
        </div>
        
        <div className="flex items-center col-span-2">
          <div className="flex items-center mr-6">
            <input 
              type="checkbox" 
              id="enforce_eager" 
              name="enforce_eager"
              className="mr-2"
              checked={formData.enforce_eager || false}
              onChange={handleChange}
            />
            <label htmlFor="enforce_eager" className="text-sm font-medium">Enforce Eager Mode</label>
          </div>
          
          <div className="flex items-center">
            <input 
              type="checkbox" 
              id="trust_remote_code" 
              name="trust_remote_code"
              className="mr-2"
              checked={formData.trust_remote_code || false}
              onChange={handleChange}
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
          
          {Object.keys(formData.additional_params).length === 0 ? (
            <div className="bg-gray-700 border border-gray-600 rounded-md p-4 text-sm text-gray-400">
              No additional parameters. Click "Add Parameter" to add custom vLLM parameters.
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(formData.additional_params).map(([key, value]) => (
                <div key={key} className="bg-gray-700 p-2 rounded-md flex items-center">
                  <div className="flex-1">
                    <span className="text-xs font-medium text-gray-300">{key}</span>
                    <span className="text-xs text-gray-400 ml-2">{value.toString()}</span>
                  </div>
                  <button 
                    type="button"
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
          type="button"
          onClick={onCancel}
          className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
        >
          Cancel
        </button>
        <button 
          type="submit"
          className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded-md flex items-center"
          disabled={!formData.name || !formData.model_id}
        >
          <Save size={16} className="mr-2" />
          {isEditMode ? 'Update Model' : 'Save Model'}
        </button>
      </div>
    </form>
  );
};

export default ModelForm;
