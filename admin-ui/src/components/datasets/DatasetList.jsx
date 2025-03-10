import React from 'react';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import { formatFileSize, formatTimestamp, formatNumber } from '../../utils/formatters';
import { Info, Upload } from 'lucide-react';

/**
 * List of available datasets
 */
const DatasetList = ({ 
  datasets, 
  loading, 
  onDelete, 
  onTrain, 
  onUpload, 
  onPreview 
}) => {
  return (
    <Card
      title="Available Datasets"
      className="overflow-hidden"
    >
      {loading ? (
        <div className="p-6 text-center">
          <LoadingSpinner message="Loading datasets..." />
        </div>
      ) : datasets.length === 0 ? (
        <div className="p-8 text-center">
          <Info size={36} className="mx-auto mb-2 text-gray-400" />
          <p className="text-gray-400 mb-4">No datasets available</p>
          <button 
            onClick={onUpload}
            className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded"
          >
            <Upload size={16} className="mr-2 inline-block" />
            Upload Your First Dataset
          </button>
        </div>
      ) : (
        <table className="min-w-full divide-y divide-gray-700">
          <thead className="bg-gray-700">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Name</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Samples</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Size</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Created</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {datasets.map(dataset => (
              <tr key={dataset.name} className="hover:bg-gray-750">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium">{dataset.name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                  {dataset.samples >= 0 ? formatNumber(dataset.samples) : 'Unknown'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                  {formatFileSize(dataset.size)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                  {formatTimestamp(dataset.created)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <button 
                    onClick={() => onPreview(dataset)}
                    className="text-blue-400 hover:text-blue-300 mr-3"
                  >
                    Preview
                  </button>
                  <button 
                    onClick={() => onTrain(dataset)}
                    className="text-purple-400 hover:text-purple-300 mr-3"
                  >
                    Train
                  </button>
                  <button 
                    onClick={() => onDelete(dataset.name)}
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

export default DatasetList;
