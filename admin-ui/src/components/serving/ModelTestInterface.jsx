import React, { useState } from 'react';
import Card from '../common/Card';
import { Play, RefreshCw } from 'lucide-react';

/**
 * Interface for testing the deployed model
 */
const ModelTestInterface = ({ 
  onTest, 
  loading, 
  response, 
  isModelReady 
}) => {
  const [prompt, setPrompt] = useState("");
  const [testParams, setTestParams] = useState({
    temperature: 0.7,
    top_p: 0.9,
    max_tokens: 256
  });
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!prompt || !isModelReady) return;
    onTest(prompt, testParams);
  };
  
  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setTestParams(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  return (
    <Card title="Test Model">
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Prompt</label>
          <textarea 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-gray-200 h-32"
            placeholder="Enter a prompt to test..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={!isModelReady}
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
              name="temperature"
              value={testParams.temperature}
              onChange={handleParamChange}
              disabled={!isModelReady}
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
              name="top_p"
              value={testParams.top_p}
              onChange={handleParamChange}
              disabled={!isModelReady}
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">Max Tokens</label>
            <input 
              type="number" 
              className="w-full bg-gray-700 border border-gray-600 rounded-md py-1 px-2 text-gray-200"
              min="1"
              step="1"
              name="max_tokens"
              value={testParams.max_tokens}
              onChange={handleParamChange}
              disabled={!isModelReady}
            />
          </div>
        </div>
        
        <button
          type="submit"
          disabled={loading || !prompt || !isModelReady}
          className={`w-full ${
            isModelReady 
              ? 'bg-purple-700 hover:bg-purple-600' 
              : 'bg-gray-600'
          } text-white py-2 rounded flex items-center justify-center ${
            (loading || !prompt || !isModelReady) ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? (
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
      </form>
      
      {response && (
        <div>
          <label className="block text-sm font-medium mb-1">Response</label>
          <div className="bg-gray-750 border border-gray-700 rounded-md p-3 text-gray-300 text-sm max-h-64 overflow-y-auto">
            {response.split('\n').map((line, i) => (
              <div key={i}>{line || <br />}</div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
};

export default ModelTestInterface;
