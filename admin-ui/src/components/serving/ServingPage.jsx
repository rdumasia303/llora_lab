import React, { useState, useEffect, useCallback, useMemo } from 'react';
import DeployModelForm from './DeployModelForm';
import ServingJobDetails from './ServingJobDetails';
import ModelTestInterface from './ModelTestInterface';
import LoadingSpinner from '../common/LoadingSpinner';
import Card from '../common/Card';
import { Terminal, RefreshCw } from 'lucide-react';
import { useJobs } from '../../hooks/useJobs';
import { useModels } from '../../hooks/useModels';
import { useAdapters } from '../../hooks/useAdapters';
import { useSystemStats } from '../../hooks/useSystemStats';

/**
 * Model serving management page
 */
const ServingPage = ({ 
  onSetActiveTab, 
  setSelectedJobLogs, 
  setError, 
  showConfirmation 
}) => {
  const [testResponse, setTestResponse] = useState("");
  const [testLoading, setTestLoading] = useState(false);
  const [logContent, setLogContent] = useState("");
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  
  const { 
    servingJobs, 
    loading, 
    error,
    logsLoading,
    fetchJobs,
    fetchJobLogs,
    startServingJob,
    stopServingJob,
    testModel,
    getActiveServingJob
  } = useJobs();
  
  const {
    models,
    fetchModels
  } = useModels();
  
  const {
    adapters,
    fetchAdapters
  } = useAdapters();
  
  const {
    systemStats,
    fetchSystemStats
  } = useSystemStats();
  
  // Memoize active serving job to prevent recalculation on every render
  const activeServingJob = useMemo(() => getActiveServingJob(), [servingJobs, getActiveServingJob]);
  
  // Initial data load - only run once
  useEffect(() => {
    // Only load initial data once
    if (initialLoadComplete) return;

    const loadInitialData = async () => {
      try {
        await Promise.all([
          fetchJobs(),
          fetchModels(),
          fetchAdapters(),
          fetchSystemStats()
        ]);
        setInitialLoadComplete(true);
      } catch (err) {
        console.error("Error loading serving data:", err);
        setError("Failed to load serving data. Please try again.");
      }
    };
    
    loadInitialData();
  }, [fetchJobs, fetchModels, fetchAdapters, fetchSystemStats, setError, initialLoadComplete]);
  
  // Set up refresh intervals ONLY when a model is deployed
  useEffect(() => {
    // Don't set up intervals if no active job or initial load isn't complete
    if (!activeServingJob || !initialLoadComplete) return;
    
    console.log("Setting up refresh intervals for active job");
    
    // Fetch logs once immediately
    fetchJobLogs(activeServingJob.id, 'serving')
      .then(result => {
        if (result && result.logs) {
          setLogContent(result.logs);
        }
      })
      .catch(console.error);
    
    // Set up refresh intervals
    const jobsInterval = setInterval(() => {
      fetchJobs().catch(console.error);
    }, 5000); // Increased from 3000ms to 5000ms
    
    const statsInterval = setInterval(() => {
      fetchSystemStats().catch(console.error);
    }, 5000);
    
    const logsInterval = setInterval(() => {
      fetchJobLogs(activeServingJob.id, 'serving')
        .then(result => {
          if (result && result.logs) {
            setLogContent(result.logs);
          }
        })
        .catch(console.error);
    }, 5000);
    
    // Clean up all intervals when component unmounts or active job changes
    return () => {
      clearInterval(jobsInterval);
      clearInterval(statsInterval);
      clearInterval(logsInterval);
    };
  }, [activeServingJob, fetchJobs, fetchJobLogs, fetchSystemStats, initialLoadComplete]);
  
  // Set error from hooks to parent
  useEffect(() => {
    if (error) setError(error);
  }, [error, setError]);
  
  // Handle deploy model
  const handleDeploy = useCallback(async (modelName, adapter = null) => {
    try {
      await startServingJob(modelName, adapter);
      
      // Clear test response
      setTestResponse("");
      
      // Refresh jobs to show the new job
      await fetchJobs();
    } catch (err) {
      console.error("Error deploying model:", err);
      setError(`Failed to deploy model: ${err.message}`);
    }
  }, [startServingJob, fetchJobs, setError]);
  
  // Handle stop serving
  const handleStopServing = useCallback(() => {
    if (!activeServingJob) return;
    
    showConfirmation({
      title: "Stop Model Serving",
      message: `Are you sure you want to stop serving the model "${activeServingJob.model_conf}"?`,
      variant: "warning",
      onConfirm: async () => {
        try {
          await stopServingJob(activeServingJob.id);
          
          // Clear test response
          setTestResponse("");
          
          // Refresh jobs
          await fetchJobs();
        } catch (err) {
          console.error("Error stopping serving:", err);
          setError(`Failed to stop serving: ${err.message}`);
        }
      }
    });
  }, [activeServingJob, stopServingJob, fetchJobs, showConfirmation, setError]);
  
  // Handle view logs
  const handleViewLogs = useCallback(() => {
    if (!activeServingJob) return;
    
    setSelectedJobLogs({
      id: activeServingJob.id,
      type: 'serving'
    });
    onSetActiveTab('logs');
  }, [activeServingJob, setSelectedJobLogs, onSetActiveTab]);
  
  // Handle test model
  const handleTestModel = useCallback(async (prompt, params) => {
    if (!prompt || !activeServingJob) return;
    
    setTestLoading(true);
    setTestResponse("");
    
    try {
      const result = await testModel(prompt, params);
      setTestResponse(result.response);
    } catch (err) {
      console.error("Error testing model:", err);
      setTestResponse(`Error: ${err.message}`);
    } finally {
      setTestLoading(false);
    }
  }, [activeServingJob, testModel]);

  // Manual refresh handler - doesn't set up intervals
  const handleManualRefresh = useCallback(async () => {
    try {
      await fetchJobs();
      if (activeServingJob) {
        await fetchSystemStats();
        const result = await fetchJobLogs(activeServingJob.id, 'serving');
        if (result && result.logs) {
          setLogContent(result.logs);
        }
      }
    } catch (err) {
      console.error("Error refreshing data:", err);
    }
  }, [fetchJobs, fetchSystemStats, fetchJobLogs, activeServingJob]);

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Model Serving</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleManualRefresh}
            className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
            disabled={loading}
          >
            <RefreshCw size={14} className={`mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh Status
          </button>
        </div>
      </div>
      
      {loading && !initialLoadComplete ? (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center">
          <LoadingSpinner message="Loading serving status..." />
        </div>
      ) : activeServingJob ? (
        <>
          <ServingJobDetails 
            job={activeServingJob}
            onViewLogs={handleViewLogs}
            onStopServing={handleStopServing}
          />
          
          <div className="mt-6">
            <ModelTestInterface 
              onTest={handleTestModel}
              loading={testLoading}
              response={testResponse}
              isModelReady={activeServingJob.status === 'ready'}
            />
          </div>
          
          <div className="mt-6">
            <Card title="Serving Logs">
              <div className="bg-gray-750 px-4 py-2 flex justify-between items-center">
                <span className="text-sm font-medium">
                  Logs for {activeServingJob.model_conf} {activeServingJob.adapter && `(${activeServingJob.adapter})`}
                </span>
                <button 
                  onClick={handleViewLogs}
                  className="text-gray-400 hover:text-white"
                >
                  <Terminal size={14} />
                </button>
              </div>
              <div className="p-4 max-h-64 overflow-y-auto bg-gray-900 font-mono text-xs">
                {logsLoading ? (
                  <div className="text-center py-4">
                    <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-purple-500" />
                    <p>Loading logs...</p>
                  </div>
                ) : logContent ? (
                  logContent.split('\n').map((line, i) => (
                    <div key={i} className="text-gray-300">{line || <br />}</div>
                  ))
                ) : (
                  <p className="text-gray-500">No logs available yet.</p>
                )}
              </div>
            </Card>
          </div>
        </>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <DeployModelForm 
              models={models}
              adapters={adapters}
              onDeploy={handleDeploy}
              loading={loading}
              systemStats={systemStats}
            />
          </div>
          
          <Card title="Resources">
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium text-gray-400 mb-2">System Resources</h4>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>GPU Memory</span>
                      <span>{systemStats.gpu.memory}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2.5">
                      <div 
                        className="bg-purple-600 h-2.5 rounded-full" 
                        style={{
                          width: `${parseFloat(systemStats.gpu.utilized || 0) * 100}%`
                        }}
                      ></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>GPU Temperature</span>
                      <span>{systemStats.gpu.temperature}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-gray-750 p-3 rounded-lg">
                <h4 className="text-sm font-medium mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  Usage Information
                </h4>
                <p className="text-xs">
                  When a model is deployed, it exposes a compatible OpenAI API endpoint 
                  at <code className="bg-gray-800 px-1 rounded">http://localhost:8000/v1</code> and 
                  a web UI at <code className="bg-gray-800 px-1 rounded">http://localhost:3000</code>.
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default ServingPage;