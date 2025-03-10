import React from 'react';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import { Info } from 'lucide-react';

/**
 * List of model configurations
 */
const ModelList = ({ 
  models, 
  loading, 
  onEdit, 
  onDelete, 
  onDeploy, 
  onAddModel 
}) => {
  return (
    <Card
      title="Available Models"
      className="overflow-hidden"
    >
      {loading ? (
        <div className="p-6 text-center">
          <LoadingSpinner message="Loading models..." />
        </div>
      ) : models.length === 0 ? (
        <div className="p-8 text-center">
          <Info size={36} className="mx-auto mb-2 text-gray-400" />
          <p className="text-gray-400 mb-4">No models configured yet</p>
          <button 
            onClick={onAddModel}
            className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded"
          >
            Add Your First Model
          </button>
        </div>
      ) : (
        <div className="overflow-x-auto">
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
                      onClick={() => onDeploy(model.name)}
                      className="text-blue-400 hover:text-blue-300 mr-3"
                    >
                      Deploy
                    </button>
                    <button 
                      onClick={() => onEdit(model.name)}
                      className="text-purple-400 hover:text-purple-300 mr-3"
                    >
                      Edit
                    </button>
                    <button 
                      onClick={() => onDelete(model.name)}
                      className="text-red-400 hover:text-red-300"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
};

export default ModelList;
