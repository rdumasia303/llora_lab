import React, { useState, useEffect } from 'react';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import { ChevronLeft, ChevronRight, X } from 'lucide-react';
import { formatFileSize, formatNumber } from '../../utils/formatters';

/**
 * Dataset preview component
 */
const DatasetPreview = ({ 
  dataset, 
  onClose, 
  samples = [],
  loading,
  onFetchSamples
}) => {
  const [currentSample, setCurrentSample] = useState(0);
  const [error, setError] = useState(null);
  
  // Reset current sample when dataset changes
  useEffect(() => {
    setCurrentSample(0);
  }, [dataset?.name]);
  
  // Fetch samples when dataset changes
  useEffect(() => {
    if (dataset && onFetchSamples) {
      try {
        onFetchSamples(dataset.name);
      } catch (err) {
        console.error("Error fetching samples:", err);
        setError("Failed to load dataset samples");
      }
    }
  }, [dataset, onFetchSamples]);
  
  if (!dataset) return null;

  return (
    <Card 
      title={`Dataset Preview: ${dataset.name}`}
      headerAction={
        <button 
          onClick={onClose}
          className="text-gray-400 hover:text-gray-200"
        >
          <X size={20} />
        </button>
      }
    >
      {error ? (
        <div className="p-6 text-center text-red-400">
          {error}
        </div>
      ) : loading ? (
        <div className="p-6 text-center">
          <LoadingSpinner message="Loading samples..." />
        </div>
      ) : samples.length === 0 ? (
        <div className="p-6 text-center text-gray-400">
          No samples available to preview.
        </div>
      ) : (
        <>
          <div className="mb-4 p-4 bg-gray-750 rounded-lg">
            <div className="flex justify-between items-center mb-2">
              <h4 className="text-sm font-medium">Sample {currentSample + 1} of {samples.length}</h4>
              <div className="flex space-x-2">
                <button 
                  onClick={() => setCurrentSample(prev => Math.max(0, prev - 1))}
                  disabled={currentSample === 0}
                  className={`p-1 rounded ${
                    currentSample === 0 ? 'text-gray-500' : 'text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <ChevronLeft size={16} />
                </button>
                <button 
                  onClick={() => setCurrentSample(prev => Math.min(samples.length - 1, prev + 1))}
                  disabled={currentSample === samples.length - 1}
                  className={`p-1 rounded ${
                    currentSample === samples.length - 1 ? 'text-gray-500' : 'text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
            
            {/* Show current sample */}
            <pre className="bg-gray-800 p-3 rounded-lg overflow-x-auto text-xs">
              {samples && samples[currentSample] ? 
                JSON.stringify(samples[currentSample], null, 2) : 
                "No data available"
              }
            </pre>
          </div>
          
          <div className="text-sm">
            <div className="mb-2">
              <span className="font-medium">Dataset Size:</span> {formatFileSize(dataset.size)}
            </div>
            <div className="mb-2">
              <span className="font-medium">Total Samples:</span> {dataset.samples >= 0 ? formatNumber(dataset.samples) : 'Unknown'}
            </div>
            
            <p className="text-gray-400 text-xs mt-4">
              Samples shown are the first {samples.length} entries in the dataset file.
            </p>
          </div>
        </>
      )}
    </Card>
  );
};

export default DatasetPreview;