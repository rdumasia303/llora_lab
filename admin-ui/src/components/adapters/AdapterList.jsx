import React from 'react';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import { Info } from 'lucide-react';
import { formatFileSize, formatTimestamp } from '../../utils/formatters';

/**
 * List of available adapters
 */
const AdapterList = ({ 
  adapters, 
  loading, 
  onDelete, 
  onDeploy, 
  onTest, 
  onCreateAdapter 
}) => {
  return (
    <Card
      title="Available Adapters"
      className="overflow-hidden"
    >
      {loading ? (
        <div className="p-6 text-center">
          <LoadingSpinner message="Loading adapters..." />
        </div>
      ) : adapters.length === 0 ? (
        <div className="p-8 text-center">
          <Info size={36} className="mx-auto mb-2 text-gray-400" />
          <p className="text-gray-400 mb-4">No adapters available</p>
          <button 
            onClick={onCreateAdapter}
            className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded"
          >
            Create Your First Adapter
          </button>
        </div>
      ) : (
        <table className="min-w-full divide-y divide-gray-700">
          <thead className="bg-gray-700">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Name</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Base Model</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Size</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Created</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {adapters.map(adapter => (
              <tr key={adapter.name} className="hover:bg-gray-750">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium">{adapter.name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                  {adapter.base_model || 'Unknown'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                  {formatFileSize(adapter.size)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                  {formatTimestamp(adapter.created)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <button 
                    onClick={() => onDeploy(adapter)}
                    className="text-blue-400 hover:text-blue-300 mr-3"
                  >
                    Deploy
                  </button>
                  <button
                    onClick={() => onTest(adapter.name)}
                    className="text-purple-400 hover:text-purple-300 mr-3"
                  >
                    Test
                  </button>
                  <button 
                    onClick={() => onDelete(adapter.name)}
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
    </Card>
  );
};

export default AdapterList;
