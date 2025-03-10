import React, { useState, useEffect } from 'react';
import Layout from './components/layout/Layout';
import Overview from './components/dashboard/Overview';
import ModelsPage from './components/models/ModelsPage';
import AdaptersPage from './components/adapters/AdaptersPage';
import DatasetsPage from './components/datasets/DatasetsPage';
import TrainingPage from './components/training/TrainingPage';
import ServingPage from './components/serving/ServingPage';
import LogsPage from './components/logs/LogsPage';
import ConfirmDialog from './components/common/ConfirmDialog';
import { useAppState } from './hooks/useAppState';
import { useSystemStats } from './hooks/useSystemStats';
import { useJobs } from './hooks/useJobs';
import { useModels } from './hooks/useModels';
import { useAdapters } from './hooks/useAdapters';
import { useDatasets } from './hooks/useDatasets';
import { TABS } from './utils/constants';

function App() {
  // App state management
  const {
    activeTab,
    setActiveTab,
    error,
    setError,
    clearError,
    showConfirmDialog,
    setShowConfirmDialog,
    confirmDialogProps,
    showConfirmation,
    closeConfirmDialog,
    selectedJobLogs,
    setSelectedJobLogs
  } = useAppState();
  
  // Core data hooks
  const { systemStats, systemMetrics, loading: statsLoading, fetchSystemStats, fetchSystemMetrics } = useSystemStats();
  const { trainingJobs, servingJobs, loading: jobsLoading, fetchJobs, startServingJob, stopServingJob, stopTrainingJob } = useJobs();
  const { models, loading: modelsLoading, fetchModels } = useModels();
  const { adapters, loading: adaptersLoading, fetchAdapters } = useAdapters();
  const { datasets, loading: datasetsLoading, fetchDatasets } = useDatasets();
  
  // Get active jobs
  const activeServingJob = servingJobs.find(job => 
    job.status !== 'stopped' && job.status !== 'failed'
  );
  
  const activeTrainingJob = trainingJobs.find(job => 
    job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped'
  );
  
  // Initialize data for the Overview page on load
  useEffect(() => {
    if (activeTab === TABS.OVERVIEW) {
      // Initial data fetch for Overview
      const fetchInitialData = async () => {
        try {
          await Promise.all([
            fetchModels(),
            fetchAdapters(),
            fetchDatasets(),
            fetchJobs(),
            fetchSystemStats(),
            fetchSystemMetrics()
          ]);
        } catch (err) {
          console.error("Error fetching initial data:", err);
          setError("Failed to load initial data. Please try again.");
        }
      };
      
      fetchInitialData();
      
      // Setup intervals for automatic refresh
      const jobsInterval = setInterval(fetchJobs, 5000);
      const statsInterval = setInterval(fetchSystemStats, 5000);
      const metricsInterval = setInterval(fetchSystemMetrics, 3000);
      
      return () => {
        clearInterval(jobsInterval);
        clearInterval(statsInterval);
        clearInterval(metricsInterval);
      };
    }
  }, [activeTab, fetchModels, fetchAdapters, fetchDatasets, fetchJobs, fetchSystemStats, fetchSystemMetrics, setError]);
  
  // Handle tab navigation
  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };
  
  // Handle stop serving
  const handleStopServing = (jobId) => {
    showConfirmation({
      title: "Stop Model Serving",
      message: "Are you sure you want to stop the currently running model?",
      variant: "warning",
      onConfirm: async () => {
        try {
          await stopServingJob(jobId);
          fetchJobs();
        } catch (err) {
          setError(`Failed to stop model: ${err.message}`);
        }
      }
    });
  };
  
  // Handle stop training
  const handleStopTraining = (jobId) => {
    showConfirmation({
      title: "Stop Training Job",
      message: "Are you sure you want to stop this training job? Progress will be lost.",
      variant: "warning",
      onConfirm: async () => {
        try {
          await stopTrainingJob(jobId);
          fetchJobs();
        } catch (err) {
          setError(`Failed to stop training: ${err.message}`);
        }
      }
    });
  };

  return (
    <>
      <Layout 
        activeTab={activeTab} 
        onTabChange={handleTabChange}
        error={error}
        onClearError={clearError}
      >
        {activeTab === TABS.OVERVIEW && (
          <Overview 
            models={models}
            adapters={adapters}
            datasets={datasets}
            trainingJobs={trainingJobs}
            servingJobs={servingJobs}
            systemStats={systemStats}
            systemMetrics={systemMetrics}
            loading={statsLoading || jobsLoading || modelsLoading || adaptersLoading || datasetsLoading}
            onNavigate={handleTabChange}
            onStopServing={handleStopServing}
            onStopTraining={handleStopTraining}
          />
        )}
        
        {activeTab === TABS.MODELS && (
          <ModelsPage 
            onSetActiveTab={setActiveTab}
            setError={setError}
            showConfirmation={showConfirmation}
          />
        )}
        
        {activeTab === TABS.ADAPTERS && (
          <AdaptersPage 
            activeServingJob={activeServingJob}
            onSetActiveTab={setActiveTab}
            setError={setError}
            showConfirmation={showConfirmation}
            closeConfirmDialog={closeConfirmDialog}
          />
        )}
        
        {activeTab === TABS.DATASETS && (
          <DatasetsPage 
            onSetActiveTab={setActiveTab}
            setError={setError}
            showConfirmation={showConfirmation}
          />
        )}
        
        {activeTab === TABS.TRAINING && (
          <TrainingPage 
            onSetActiveTab={setActiveTab}
            setSelectedJobLogs={setSelectedJobLogs}
            setError={setError}
            showConfirmation={showConfirmation}
          />
        )}
        
        {activeTab === TABS.SERVING && (
          <ServingPage 
            onSetActiveTab={setActiveTab}
            setSelectedJobLogs={setSelectedJobLogs}
            setError={setError}
            showConfirmation={showConfirmation}
          />
        )}
        
        {activeTab === TABS.LOGS && (
          <LogsPage 
            selectedJobLogs={selectedJobLogs}
            setError={setError}
          />
        )}
      </Layout>
      
      {/* Global confirm dialog */}
      <ConfirmDialog 
        isOpen={showConfirmDialog} 
        onClose={closeConfirmDialog}
        {...confirmDialogProps}
      />
    </>
  );
}

export default App;
