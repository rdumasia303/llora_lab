import React, { useState } from 'react';
import Card from '../common/Card';
import { Upload, Info } from 'lucide-react';
import LoadingSpinner from '../common/LoadingSpinner';

/**
 * Dataset upload form
 */
const DatasetUpload = ({ 
  onUpload, 
  onCancel, 
  loading 
}) => {
  const [uploadFile, setUploadFile] = useState(null);
  
  const handleFileChange = (e) => {
    setUploadFile(e.target.files[0]);
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!uploadFile) return;
    
    onUpload(uploadFile);
  };

  return (
    <Card title="Upload Dataset">
      <form onSubmit={handleSubmit}>
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Select JSONL File</label>
          <input
            type="file"
            id="datasetUpload"
            accept=".jsonl"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-400
              file:mr-4 file:py-2 file:px-4
              file:rounded file:border-0
              file:text-sm file:font-semibold
              file:bg-gray-700 file:text-gray-200
              hover:file:bg-gray-600"
          />
          <p className="mt-2 text-xs text-gray-400">
            Only JSONL format is supported. Each line must be a valid JSON object.
          </p>
        </div>
        
        <div className="bg-gray-750 p-4 rounded-lg mb-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center">
            <Info size={14} className="mr-1" />
            Dataset Format
          </h4>
          <p className="text-xs mb-2">
            Datasets should be in JSONL format with each line containing a training sample.
            The expected format depends on the model and training approach:
          </p>
          <pre className="bg-gray-800 p-2 rounded text-xs overflow-x-auto">
            {"// Example for chat completion:\n" +
            '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there! How can I help you today?"}]}\n' +
            "\n// Example for text completion:\n" +
            '{"text": "Once upon a time, there was a little cottage in the woods."}'
            }
          </pre>
        </div>
        
        <div className="flex justify-end space-x-2">
          <button 
            type="button"
            onClick={onCancel}
            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
            disabled={loading}
          >
            Cancel
          </button>
          <button 
            type="submit"
            className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded-md flex items-center"
            disabled={!uploadFile || loading}
          >
            {loading ? (
              <>
                <LoadingSpinner size="small" className="mr-2" /> 
                Uploading...
              </>
            ) : (
              <>
                <Upload size={16} className="mr-2" />
                Upload Dataset
              </>
            )}
          </button>
        </div>
      </form>
    </Card>
  );
};

export default DatasetUpload;
