import React, { useEffect, useMemo } from 'react';
import SystemOverview from './SystemOverview';
import SystemMetrics from './SystemMetrics';
import ResourceStats from './ResourceStats';
import LoadingSpinner from '../common/LoadingSpinner';

/**
 * Main overview dashboard component
 */
const Overview = ({ 
  models, 
  adapters, 
  datasets,
  trainingJobs,
  servingJobs,
  systemStats,
  systemMetrics,
  loading,
  onNavigate,
  onStopServing,
  onStopTraining
}) => {
  // Get active jobs
  const activeServingJob = useMemo(() => 
    servingJobs.find(job => 
      job.status !== 'stopped' && job.status !== 'failed'
    ), [servingJobs]);
  
  const activeTrainingJob = useMemo(() => 
    trainingJobs.find(job => 
      job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped'
    ), [trainingJobs]);
  
  // Prepare chart data
  const systemMetricsData = useMemo(() => {
    return systemMetrics.timestamps.map((timestamp, index) => ({
      timestamp,
      gpu: systemMetrics.gpu_utilization[index] || 0,
      memory: systemMetrics.memory_usage[index] || 0
    }));
  }, [systemMetrics]);
  
  const trainingChartData = useMemo(() => {
    if (!activeTrainingJob || activeTrainingJob.step <= 0) return [];
    
    return Array.from({ length: activeTrainingJob.step }, (_, i) => ({
      step: i + 1,
      loss: activeTrainingJob.loss 
        ? (activeTrainingJob.loss - (0.01 * (activeTrainingJob.step - (i + 1)))) 
        : null,
      learningRate: activeTrainingJob.learning_rate
    }));
  }, [activeTrainingJob]);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold mb-4">System Overview</h2>
      
      {loading && !systemStats.containers ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="large" message="Loading system overview..." />
        </div>
      ) : (
        <>
          <SystemOverview 
            systemStats={systemStats} 
            activeServingJob={activeServingJob}
            activeTrainingJob={activeTrainingJob}
            onNavigate={onNavigate}
            onStopServing={onStopServing}
            onStopTraining={onStopTraining}
          />
          
          <SystemMetrics 
            systemMetricsData={systemMetricsData}
            trainingChartData={trainingChartData}
            loading={loading}
          />
          
          <ResourceStats 
            models={models}
            adapters={adapters}
            datasets={datasets}
            jobs={trainingJobs.length + servingJobs.length}
            containers={systemStats.containers || []}
            onNavigate={onNavigate}
          />
        </>
      )}
    </div>
  );
};

export default Overview;
