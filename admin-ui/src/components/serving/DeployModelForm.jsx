import React, { useState, useEffect } from 'react';
import Card from '../common/Card';
import { formatPercent } from '../../utils/formatters';
import { Play, RefreshCw } from 'lucide-react';

/**
 * Form for deploying a model with an optional adapter
 */
const DeployModelForm = ({ 
  models, 
  adapters, 
  onDeploy, 
  loading,
  systemStats
}) => {
  const [formData, setFormData] = useState({
    model: "",
    adapter: ""
  });
  
  // Filter adapters based on selected model
  const compatibleAdapters = formData.model ? 
    adapters.filter(adapter => adapter.base_model === formData.model) : 
    [];
  
  // Handle form changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // If changing model, reset adapter selection
    if (name === 'model') {
      setFormData({
        model: value,
        adapter: ""
      });
    } else {
      setFormData({
        ...formData,
        [name]: value
      });
    }
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.model) return;
    
    onDeploy(formData.model, formData.adapter || null);
  };
  
  // Try to load deployment information from session storage (from other pages)
  useEffect(() => {
    const storedModel = window.sessionStorage.getItem('deploy_model');
    const storedAdapter = window.sessionStorage.getItem('deploy_adapter');
    
    if (storedModel) {
      setFormData({
        model: storedModel,
        adapter: storedAdapter || ""
      });
      
      // Clear the storage
      window.sessionStorage.removeItem('deploy_model');
      window.sessionStorage.removeItem('deploy_adapter');
    }
  }, []);

  return (
    <Card title="Deploy a Model">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium mb-1">Model</label>
            <select
              className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200"
              value={formData.model}
              name="model"
              onChange={handleChange}
              required
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
              value={formData.adapter}
              name="adapter"
              onChange={handleChange}
              disabled={!formData.model}
            >
              <option value="">No adapter</option>
              {compatibleAdapters.map(adapter => (
                <option key={adapter.name} value={adapter.name}>{adapter.name}</option>
              ))}
            </select>
            {formData.model && compatibleAdapters.length === 0 && (
              <p className="text-xs text-yellow-400 mt-1">
                No compatible adapters found for this model.
              </p>
            )}
          </div>
        </div>
        
        {formData.model && (
          <div className="bg-gray-750 p-4 rounded-lg mb-6">
            <h4 className="text-sm font-medium text-gray-400 mb-2">Selected Model Details</h4>
            {models.filter(m => m.name === formData.model).map(model => (
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
                  {model.tensor_parallel_size && model.tensor_parallel_size > 1 && (
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
          type="submit"
          disabled={!formData.model || loading}
          className={`bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded flex items-center ${
            !formData.model || loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? (
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
      </form>
    </Card>
  );
};

export default DeployModelForm;
