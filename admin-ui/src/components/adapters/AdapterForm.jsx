import React, { useState, useEffect } from 'react';
import { MODEL_CONFIG_OPTIONS } from '../../utils/constants';
import { validateAdapterForm } from '../../utils/validators';
import { Save, Play } from 'lucide-react';

/**
 * Form for creating new adapters and starting training
 */
const AdapterForm = ({ 
  models, 
  datasets, 
  onSave, 
  onStartTraining, 
  onCancel 
}) => {
  const [formData, setFormData] = useState({
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
  
  const [errors, setErrors] = useState({});
  
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
                    name === 'learning_rate' || name === 'lora_dropout' ? 
                    parseFloat(value) : parseInt(value);
    
    setFormData(prev => ({
      ...prev,
      [name]: parsedValue
    }));
  };
  
  // Form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate form
    const validation = validateAdapterForm(formData);
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
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium mb-1">Name</label>
          <input 
            type="text" 
            className={`w-full bg-gray-700 border ${errors.name ? 'border-red-500' : 'border-gray-600'} rounded-md py-2 px-3 text-gray-200`}
            value={formData.name}
            name="name"
            onChange={handleChange}
            placeholder="e.g., my-custom-lora"
          />
          {errors.name && <p className="text-xs text-red-400 mt-1">{errors.name}</p>}
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Base Model</label>
          <select
            className={`w-full bg-gray-700 border ${errors.base_model ? 'border-red-500' : 'border-gray-600'} rounded-md py-2 px-3 text-gray-200`}
            value={formData.base_model}
            name="base_model"
            onChange={handleChange}
          >
            <option value="">Select a model</option>
            {models.map(model => (
              <option key={model.name} value={model.name}>{model.name}</option>
            ))}
          </select>
          {errors.base_model && (
            <p className="text-xs text-red-400 mt-1">{errors.base_model}</p>
          )}
          {models.length === 0 && (
            <p className="text-xs text-yellow-400 mt-1">
              No models available. Please add a model first.
            </p>
          )}
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Dataset</label>
          <select
            className={`w-full bg-gray-700 border ${errors.dataset ? 'border-red-500' : 'border-gray-600'} rounded-md py-2 px-3 text-gray-200`}
            value={formData.dataset}
            name="dataset"
            onChange={handleChange}
          >
            <option value="">Select a dataset</option>
            {datasets.map(dataset => (
              <option key={dataset.name} value={dataset.name}>{dataset.name}</option>
            ))}
          </select>
          {errors.dataset && (
            <p className="text-xs text-red-400 mt-1">{errors.dataset}</p>
          )}
          {datasets.length === 0 && (
            <p className="text-xs text-yellow-400 mt-1">
              No datasets available. Please upload a dataset first.
            </p>
          )}
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Description</label>
          <input 
            type="text" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.description || ''}
            name="description"
            onChange={handleChange}
            placeholder="Optional description"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">LoRA Rank</label>
          <input 
            type="number" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.lora_rank || ''}
            name="lora_rank"
            onChange={handleNumberChange}
            placeholder="e.g., 8"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">LoRA Alpha</label>
          <input 
            type="number" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.lora_alpha || ''}
            name="lora_alpha"
            onChange={handleNumberChange}
            placeholder="e.g., 16"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">LoRA Dropout</label>
          <input 
            type="number" 
            step="0.01"
            min="0"
            max="1"
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.lora_dropout || ''}
            name="lora_dropout"
            onChange={handleNumberChange}
            placeholder="e.g., 0.05"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Training Steps</label>
          <input 
            type="number" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.steps || ''}
            name="steps"
            onChange={handleNumberChange}
            placeholder="e.g., 100"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Batch Size</label>
          <input 
            type="number" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.batch_size || ''}
            name="batch_size"
            onChange={handleNumberChange}
            placeholder="e.g., 8"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Learning Rate</label>
          <input 
            type="number" 
            step="0.0001"
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.learning_rate || ''}
            name="learning_rate"
            onChange={handleNumberChange}
            placeholder="e.g., 0.0002"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Max Sequence Length</label>
          <input 
            type="number" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.max_seq_length || ''}
            name="max_seq_length"
            onChange={handleNumberChange}
            placeholder="e.g., 2048"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Chat Template (Optional)</label>
          <select 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
            value={formData.chat_template || ''}
            name="chat_template"
            onChange={handleChange}
          >
            <option value="">Use base model template</option>
            {MODEL_CONFIG_OPTIONS.CHAT_TEMPLATE.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
        
        <div className="flex items-center">
          <input 
            type="checkbox" 
            id="use_nested_quant" 
            name="use_nested_quant"
            className="mr-2"
            checked={formData.use_nested_quant || false}
            onChange={handleChange}
          />
          <label htmlFor="use_nested_quant" className="text-sm font-medium">Use Nested Quantization</label>
        </div>
      </div>
      
      <div className="bg-gray-750 p-4 rounded-lg mb-6">
        <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
          Training Information
        </h4>
        <p className="text-xs">
          LoRA (Low-Rank Adaptation) is a fine-tuning technique that adds small trainable layers to the frozen model.
          The rank parameter determines the expressiveness of the adapter, with higher values providing more capacity
          but requiring more memory and training time.
        </p>
      </div>
      
      <div className="flex justify-end space-x-2">
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
          disabled={!formData.name || !formData.base_model || !formData.dataset}
        >
          <Save size={16} className="mr-2" />
          Save Adapter Config
        </button>
        <button 
          type="button"
          onClick={() => {
            // First validate form
            const validation = validateAdapterForm(formData);
            if (!validation.isValid) {
              setErrors(validation.errors);
              return;
            }
            
            // Then start training
            const cleanedData = Object.fromEntries(
              Object.entries(formData).filter(([_, v]) => v !== null && v !== undefined && v !== "")
            );
            
            onStartTraining(cleanedData);
          }}
          className="bg-green-700 hover:bg-green-600 text-white px-4 py-2 rounded-md flex items-center"
          disabled={!formData.name || !formData.base_model || !formData.dataset}
        >
          <Play size={16} className="mr-2" />
          Start Training
        </button>
      </div>
    </form>
  );
};

export default AdapterForm;
