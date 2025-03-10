import React, { useState, useEffect, useCallback } from 'react';
import DatasetList from './DatasetList';
import DatasetUpload from './DatasetUpload';
import DatasetPreview from './DatasetPreview';
import { useDatasets } from '../../hooks/useDatasets';
import { formatNumber } from '../../utils/formatters';

/**
 * Datasets management page
 */
const DatasetsPage = ({ 
  onSetActiveTab, 
  setError, 
  showConfirmation 
}) => {
  const [activeTab, setActiveTab] = useState('list');
  const [previewDataset, setPreviewDataset] = useState(null);
  const [sampleData, setSampleData] = useState([]);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  
  const { 
    datasets, 
    loading, 
    error, 
    fetchDatasets, 
    uploadDataset, 
    deleteDataset,
    previewDataset: fetchDatasetPreview
  } = useDatasets();
  
  // Initialize with data - only once
  useEffect(() => {
    if (initialLoadComplete) return;
    
    fetchDatasets()
      .then(() => setInitialLoadComplete(true))
      .catch(err => {
        console.error("Error loading datasets:", err);
        setError("Failed to load datasets. Please try again.");
      });
  }, [fetchDatasets, setError, initialLoadComplete]);
  
  // Set error from datasets hook to parent
  useEffect(() => {
    if (error) setError(error);
  }, [error, setError]);
  
  // Handle dataset upload
  const handleUpload = useCallback(async (file) => {
    try {
      await uploadDataset(file);
      await fetchDatasets();
      setActiveTab('list');
    } catch (err) {
      console.error("Error uploading dataset:", err);
      setError(`Failed to upload dataset: ${err.message}`);
    }
  }, [uploadDataset, fetchDatasets, setError]);
  
  // Handle dataset deletion
  const handleDelete = useCallback((name) => {
    showConfirmation({
      title: "Delete Dataset",
      message: `Are you sure you want to delete dataset "${name}"?`,
      variant: "danger",
      onConfirm: async () => {
        try {
          await deleteDataset(name);
          await fetchDatasets();
          
          // Close preview if the deleted dataset was being previewed
          if (previewDataset && previewDataset.name === name) {
            setPreviewDataset(null);
            setSampleData([]);
          }
        } catch (err) {
          console.error("Error deleting dataset:", err);
          setError(`Failed to delete dataset: ${err.message}`);
        }
      }
    });
  }, [deleteDataset, fetchDatasets, setError, showConfirmation, previewDataset]);
  
  // Handle training with a dataset
  const handleTrain = useCallback((dataset) => {
    // Navigate to adapters/create tab with this dataset pre-selected
    onSetActiveTab('adapters');
    // Store selection in session storage for the adapters page to pick up
    window.sessionStorage.setItem('selected_dataset', dataset.name);
  }, [onSetActiveTab]);
  
  // Handle dataset preview
  const handlePreview = useCallback((dataset) => {
    // Clear previous sample data when switching datasets
    setSampleData([]);
    setPreviewDataset(dataset);
  }, []);
  
  // Fetch dataset samples for preview - safer implementation
  const fetchSamples = useCallback(async (datasetName) => {
    if (!datasetName) return;
    
    setSampleLoading(true);
    setSampleData([]); // Clear previous data
    
    try {
      const result = await fetchDatasetPreview(datasetName, 10);
      
      // Validate the response
      if (result && Array.isArray(result.samples)) {
        setSampleData(result.samples);
      } else {
        console.error("Invalid samples data format:", result);
        setSampleData([]);
        setError("Received invalid data format from server");
      }
    } catch (err) {
      console.error("Error fetching dataset samples:", err);
      setSampleData([]);
      setError(`Failed to load dataset samples: ${err.message || "Unknown error"}`);
    } finally {
      setSampleLoading(false);
    }
  }, [fetchDatasetPreview, setError]);

  // Manual refresh handler
  const handleRefresh = useCallback(async () => {
    try {
      await fetchDatasets();
      // If previewing a dataset, refresh its samples too
      if (previewDataset) {
        await fetchSamples(previewDataset.name);
      }
    } catch (err) {
      console.error("Error refreshing data:", err);
      setError("Failed to refresh data");
    }
  }, [fetchDatasets, previewDataset, fetchSamples, setError]);

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Datasets</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleRefresh}
            className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
            disabled={loading || sampleLoading}
          >
            <svg className={`mr-1 h-4 w-4 ${loading || sampleLoading ? 'animate-spin' : ''}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Refresh
          </button>
          <button 
            onClick={() => setActiveTab('upload')}
            className="bg-green-700 hover:bg-green-600 text-white px-3 py-1 rounded text-sm flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            Upload Dataset
          </button>
        </div>
      </div>
      
      {activeTab === 'list' && (
        <>
          <DatasetList 
            datasets={datasets}
            loading={loading && !initialLoadComplete}
            onDelete={handleDelete}
            onTrain={handleTrain}
            onUpload={() => setActiveTab('upload')}
            onPreview={handlePreview}
          />
          
          {previewDataset && (
            <div className="mt-6">
              <DatasetPreview 
                dataset={previewDataset}
                onClose={() => {
                  setPreviewDataset(null);
                  setSampleData([]);
                }}
                samples={sampleData}
                loading={sampleLoading}
                onFetchSamples={fetchSamples}
              />
            </div>
          )}
        </>
      )}
      
      {activeTab === 'upload' && (
        <DatasetUpload 
          onUpload={handleUpload}
          onCancel={() => setActiveTab('list')}
          loading={loading}
        />
      )}
    </div>
  );
};

export default DatasetsPage;