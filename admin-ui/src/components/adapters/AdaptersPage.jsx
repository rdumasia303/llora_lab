import React, { useState, useEffect, useCallback } from 'react';
import AdapterList from './AdapterList';
import AdapterForm from './AdapterForm';
import AdapterTestInterface from './AdapterTestInterface';
import { useAdapters } from '../../hooks/useAdapters';
import { useModels } from '../../hooks/useModels';
import { useDatasets } from '../../hooks/useDatasets';
import { useJobs } from '../../hooks/useJobs';

/**
 * Adapters management page
 */
const AdaptersPage = ({ 
  activeServingJob,
  onSetActiveTab, 
  setError, 
  showConfirmation,
  closeConfirmDialog
}) => {
  // Check if we should start on create tab based on session storage
  const initialTab = window.sessionStorage.getItem('adapters_active_tab') === 'create' ? 'create' : 'list';
  const [activeTab, setActiveTab] = useState(initialTab);
  const [selectedAdapter, setSelectedAdapter] = useState("");
  
  // Clear session storage after reading it
  useEffect(() => {
    if (window.sessionStorage.getItem('adapters_active_tab')) {
      window.sessionStorage.removeItem('adapters_active_tab');
    }
  }, []);
  
  // Hooks
  const { 
    adapters,
    loading: adaptersLoading,
    error: adaptersError,
    fetchAdapters,
    createAdapterConfig,
    deleteAdapter,
    startTraining
  } = useAdapters();
  
  const {
    models,
    loading: modelsLoading,
    fetchModels
  } = useModels();
  
  const {
    datasets,
    loading: datasetsLoading,
    fetchDatasets
  } = useDatasets();
  
  const {
    startServingJob,
    stopServingJob
  } = useJobs();
  
  // Initialize with data
  useEffect(() => {
    Promise.all([
      fetchAdapters(),
      fetchModels(),
      fetchDatasets()
    ]).catch(err => {
      console.error("Error loading adapter page data:", err);
      setError("Failed to load adapter page data. Please try again.");
    });
  }, [fetchAdapters, fetchModels, fetchDatasets, setError]);
  
  // Check for pre-selected dataset from session storage
  useEffect(() => {
    const selectedDataset = window.sessionStorage.getItem('selected_dataset');
    if (selectedDataset) {
      // Move to create tab with the dataset selected
      setActiveTab('create');
      // Remove from session storage
      window.sessionStorage.removeItem('selected_dataset');
    }
  }, [datasets]);
  
  // Set error from hooks to parent
  useEffect(() => {
    if (adaptersError) setError(adaptersError);
  }, [adaptersError, setError]);
  
  // Handle adapter creation
  const handleCreateAdapter = useCallback(async (adapterData) => {
    try {
      await createAdapterConfig(adapterData);
      await fetchAdapters();
      setActiveTab('list');
    } catch (err) {
      console.error("Error creating adapter:", err);
      setError(`Failed to create adapter: ${err.message}`);
    }
  }, [createAdapterConfig, fetchAdapters, setError]);
  
  // Handle adapter deletion
  const handleDeleteAdapter = useCallback((adapterName) => {
    showConfirmation({
      title: "Delete Adapter",
      message: `Are you sure you want to delete adapter "${adapterName}"?`,
      variant: "danger",
      onConfirm: async () => {
        try {
          await deleteAdapter(adapterName);
          await fetchAdapters();
        } catch (err) {
          console.error("Error deleting adapter:", err);
          setError(`Failed to delete adapter: ${err.message}`);
        }
      }
    });
  }, [deleteAdapter, fetchAdapters, setError, showConfirmation]);
  
  // Handle adapter deployment
  const handleDeployAdapter = useCallback((adapter) => {
    // Check if there's already an active serving job
    if (activeServingJob) {
      showConfirmation({
        title: "Stop Current Model",
        message: `There's already a model running (${activeServingJob.model_conf}). Do you want to stop it and deploy ${adapter.name} instead?`,
        variant: "warning",
        onConfirm: async () => {
          try {
            // Stop current job
            await stopServingJob(activeServingJob.id);
            
            // Wait a moment to ensure it's stopped
            setTimeout(() => {
              // Navigate to serving page with adapter selected
              onSetActiveTab('serving');
              // We can pass state through URL or another mechanism, but for simplicity:
              window.sessionStorage.setItem('deploy_adapter', adapter.name);
              window.sessionStorage.setItem('deploy_model', adapter.base_model);
            }, 1000);
          } catch (err) {
            console.error("Error stopping current model:", err);
            setError(`Failed to stop current model: ${err.message}`);
          }
        }
      });
    } else {
      // No active job, just navigate to serving page
      onSetActiveTab('serving');
      // Pass deploy information
      window.sessionStorage.setItem('deploy_adapter', adapter.name);
      window.sessionStorage.setItem('deploy_model', adapter.base_model);
    }
  }, [activeServingJob, onSetActiveTab, showConfirmation, stopServingJob, setError]);
  
  // Handle start training
  const handleStartTraining = useCallback(async (adapterData) => {
    // Check if there's an active serving job first - serving and training can't run together
    if (activeServingJob) {
      showConfirmation({
        title: "Stop Current Model",
        message: `Training requires GPU resources. There's a model running (${activeServingJob.model_conf}). Do you want to stop it before starting training?`,
        variant: "warning",
        onConfirm: async () => {
          try {
            // Stop current job
            await stopServingJob(activeServingJob.id);
            
            // Wait a moment to ensure it's stopped
            setTimeout(async () => {
              try {
                // First create the adapter config
                await createAdapterConfig(adapterData);
                // Then start training
                await startTraining(adapterData.name);
                
                // Navigate to training page
                onSetActiveTab('training');
              } catch (err) {
                console.error("Error starting training:", err);
                setError(`Failed to start training: ${err.message}`);
              }
            }, 1000);
          } catch (err) {
            console.error("Error stopping current model:", err);
            setError(`Failed to stop current model: ${err.message}`);
          }
        }
      });
    } else {
      try {
        // First create the adapter config
        await createAdapterConfig(adapterData);
        // Then start training
        await startTraining(adapterData.name);
        
        // Navigate to training page
        onSetActiveTab('training');
      } catch (err) {
        console.error("Error starting training:", err);
        setError(`Failed to start training: ${err.message}`);
      }
    }
  }, [activeServingJob, createAdapterConfig, startTraining, onSetActiveTab, showConfirmation, stopServingJob, setError]);

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Adapters</h2>
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('list')}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'list' ? 'bg-purple-700 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
          >
            Adapter List
          </button>
          <button
            onClick={() => setActiveTab('create')}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'create' ? 'bg-purple-700 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
          >
            Create Adapter
          </button>
          <button
            onClick={() => {
              // Make sure datasets are loaded before showing test interface
              Promise.all([fetchDatasets(), fetchModels(), fetchAdapters()]);
              setActiveTab('test');
            }}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'test' ? 'bg-purple-700 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
          >
            Test Adapter
          </button>
        </div>
      </div>
      
      {activeTab === 'list' && (
        <AdapterList 
          adapters={adapters}
          loading={adaptersLoading}
          onDelete={handleDeleteAdapter}
          onDeploy={handleDeployAdapter}
          onTest={(adapterName) => {
            setSelectedAdapter(adapterName);
            setActiveTab('test');
          }}
          onCreateAdapter={() => setActiveTab('create')}
        />
      )}
      
      {activeTab === 'create' && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h3 className="text-lg font-semibold mb-4">Create New Adapter</h3>
          <AdapterForm 
            models={models}
            datasets={datasets}
            onSave={handleCreateAdapter}
            onStartTraining={handleStartTraining}
            onCancel={() => setActiveTab('list')}
            // Pass pre-selected dataset if available
            preSelectedDataset={window.sessionStorage.getItem('selected_dataset')}
          />
        </div>
      )}
      
      {activeTab === 'test' && (
        <AdapterTestInterface
          adapters={adapters}
          selectedAdapter={selectedAdapter}
          onSelectAdapter={setSelectedAdapter}
          onNavigateToServing={() => onSetActiveTab('serving')}
        />
      )}
    </div>
  );
};

export default AdaptersPage;