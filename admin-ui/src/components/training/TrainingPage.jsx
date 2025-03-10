import React, { useState, useEffect, useCallback } from 'react';
import { Plus, RefreshCw } from 'lucide-react';
import ActiveTrainingJob from './ActiveTrainingJob';
import TrainingJobList from './TrainingJobList';
import LoadingSpinner from '../common/LoadingSpinner';
import { useJobs } from '../../hooks/useJobs';
import { useAdapters } from '../../hooks/useAdapters';

/**
 * Training management page
 */
const TrainingPage = ({ 
  onSetActiveTab, 
  setSelectedJobLogs, 
  setError, 
  showConfirmation 
}) => {
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  
  const { 
    trainingJobs, 
    loading, 
    error, 
    fetchJobs, 
    stopTrainingJob 
  } = useJobs();
  
  const {
    adapters,
    fetchAdapters
  } = useAdapters();
  
  // Get active job from the list
  const activeTrainingJob = trainingJobs.find(job => 
    job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped'
  );
  
  // Initialize data
  useEffect(() => {
    if (initialLoadComplete) return;
    
    Promise.all([
      fetchJobs(),
      fetchAdapters()
    ])
    .then(() => {
      setInitialLoadComplete(true);
    })
    .catch(err => {
      console.error("Error loading training data:", err);
      setError("Failed to load training data. Please try again.");
    });
  }, [fetchJobs, fetchAdapters, setError, initialLoadComplete]);
  
  // Setup refresh interval only when there's an active job
  useEffect(() => {
    if (!initialLoadComplete || !activeTrainingJob) return;
    
    console.log("Setting up refresh interval for active training job");
    
    // Setup refresh interval
    const interval = setInterval(() => {
      fetchJobs().catch(console.error);
    }, 5000); // Increased from 3000ms to reduce flickering
    
    return () => clearInterval(interval);
  }, [fetchJobs, activeTrainingJob, initialLoadComplete]);
  
  // Set error from hooks to parent
  useEffect(() => {
    if (error) setError(error);
  }, [error, setError]);
  
  // Handle view logs
  const handleViewLogs = useCallback((job) => {
    setSelectedJobLogs({
      id: job.id,
      type: 'training'
    });
    onSetActiveTab('logs');
  }, [setSelectedJobLogs, onSetActiveTab]);
  
  // Handle stop training
  const handleStopTraining = useCallback((jobId) => {
    showConfirmation({
      title: "Stop Training Job",
      message: "Are you sure you want to stop this training job? This cannot be undone.",
      variant: "warning",
      onConfirm: async () => {
        try {
          await stopTrainingJob(jobId);
          await fetchJobs();
        } catch (err) {
          console.error("Error stopping training job:", err);
          setError(`Failed to stop training job: ${err.message}`);
        }
      }
    });
  }, [stopTrainingJob, fetchJobs, setError, showConfirmation]);
  
  // Handle deploy adapter
  const handleDeploy = useCallback((job) => {
    // Get adapter details from job
    const adapterName = job.adapter_config;
    const adapter = adapters.find(a => a.name === adapterName);
    
    if (!adapter || !adapter.base_model) {
      setError("Cannot deploy adapter: missing base model information");
      return;
    }
    
    // Navigate to serving page with adapter selected
    onSetActiveTab('serving');
    // We can pass state through sessionStorage
    window.sessionStorage.setItem('deploy_adapter', adapterName);
    window.sessionStorage.setItem('deploy_model', adapter.base_model);
  }, [adapters, onSetActiveTab, setError]);
  
  // Handle refresh
  const handleRefresh = useCallback(async () => {
    try {
      await fetchJobs();
      await fetchAdapters();
    } catch (err) {
      console.error("Error refreshing data:", err);
      setError("Failed to refresh data");
    }
  }, [fetchJobs, fetchAdapters, setError]);

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Training Jobs</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleRefresh}
            className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
            disabled={loading}
          >
            <RefreshCw size={14} className={`mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh Status
          </button>
          <button 
            onClick={() => {
              onSetActiveTab('adapters');
              // Pass state to indicate creating new adapter
              window.sessionStorage.setItem('adapters_active_tab', 'create');
            }}
            className="bg-green-700 hover:bg-green-600 text-white px-3 py-1 rounded text-sm flex items-center"
          >
            <Plus size={14} className="mr-1" /> New Training Job
          </button>
        </div>
      </div>
      
      {loading && !initialLoadComplete ? (
        <div className="p-6 text-center bg-gray-800 rounded-lg border border-gray-700">
          <LoadingSpinner message="Loading training jobs..." />
        </div>
      ) : trainingJobs.length === 0 ? (
        <div className="p-8 text-center bg-gray-800 rounded-lg border border-gray-700">
          <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2 text-gray-400"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
          <p className="text-gray-400 mb-4">No training jobs found</p>
          <button 
            onClick={() => {
              onSetActiveTab('adapters');
              window.sessionStorage.setItem('adapters_active_tab', 'create');
            }}
            className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded"
          >
            Start Your First Training Job
          </button>
        </div>
      ) : (
        <>
          {activeTrainingJob && (
            <ActiveTrainingJob 
              job={activeTrainingJob}
              onViewLogs={handleViewLogs}
              onStopTraining={handleStopTraining}
            />
          )}
          
          <TrainingJobList 
            jobs={trainingJobs
              .slice()
              .sort((a, b) => new Date(b.start_time) - new Date(a.start_time))}
            onViewLogs={handleViewLogs}
            onStopTraining={handleStopTraining}
            onDeploy={handleDeploy}
          />
        </>
      )}
    </div>
  );
};

export default TrainingPage;