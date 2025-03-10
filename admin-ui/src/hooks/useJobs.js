import { useState, useCallback } from 'react';
import { useApi } from './useApi';

/**
 * Hook for managing training and serving jobs
 */
export const useJobs = () => {
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [servingJobs, setServingJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [logsLoading, setLogsLoading] = useState(false);
  const { fetchApi, error, clearError } = useApi();

  // Fetch all jobs (both training and serving)
  const fetchJobs = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch both types of jobs in parallel
      const [trainingData, servingData] = await Promise.all([
        fetchApi('/training/jobs').catch(err => {
          console.error("Error fetching training jobs:", err);
          return [];
        }),
        fetchApi('/serving/jobs').catch(err => {
          console.error("Error fetching serving jobs:", err);
          return [];
        })
      ]);
      
      setTrainingJobs(trainingData || []);
      setServingJobs(servingData || []);
      return { trainingJobs: trainingData, servingJobs: servingData };
    } catch (error) {
      console.error("Error fetching jobs:", error);
      return { trainingJobs: [], servingJobs: [] };
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Start a training job
  const startTrainingJob = useCallback(async (adapterName) => {
    setLoading(true);
    try {
      const result = await fetchApi(`/training/start?adapter_name=${adapterName}`, {
        method: 'POST'
      });
      await fetchJobs(); // Refresh job list
      return result;
    } catch (error) {
      console.error(`Error starting training for ${adapterName}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi, fetchJobs]);

  // Stop a training job
  const stopTrainingJob = useCallback(async (jobId) => {
    setLoading(true);
    try {
      const result = await fetchApi(`/training/jobs/${jobId}`, {
        method: 'DELETE'
      });
      await fetchJobs(); // Refresh job list
      return result;
    } catch (error) {
      console.error(`Error stopping training job ${jobId}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi, fetchJobs]);

  // Start a serving job
  const startServingJob = useCallback(async (modelName, adapter = null) => {
    setLoading(true);
    try {
      const url = `/serving/start?model_name=${modelName}${adapter ? `&adapter=${adapter}` : ''}`;
      const result = await fetchApi(url, {
        method: 'POST'
      });
      await fetchJobs(); // Refresh job list
      return result;
    } catch (error) {
      console.error(`Error starting serving for ${modelName}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi, fetchJobs]);

  // Stop a serving job
  const stopServingJob = useCallback(async (jobId) => {
    setLoading(true);
    try {
      const result = await fetchApi(`/serving/jobs/${jobId}`, {
        method: 'DELETE'
      });
      await fetchJobs(); // Refresh job list
      return result;
    } catch (error) {
      console.error(`Error stopping serving job ${jobId}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi, fetchJobs]);

  // Get logs for a job
  const fetchJobLogs = useCallback(async (jobId, jobType) => {
    if (!jobId) return { logs: "" };
    
    setLogsLoading(true);
    try {
      const endpoint = jobType === 'training' 
        ? `/training/logs/${jobId}`
        : `/serving/logs/${jobId}`;
        
      return await fetchApi(endpoint);
    } catch (error) {
      console.error(`Error fetching logs for ${jobType} job ${jobId}:`, error);
      return { logs: `Error fetching logs: ${error.message}` };
    } finally {
      setLogsLoading(false);
    }
  }, [fetchApi]);

  // Test a model (active serving job)
  const testModel = useCallback(async (prompt, params = {}) => {
    setLoading(true);
    try {
      const queryParams = new URLSearchParams({
        prompt: prompt,
        temperature: params.temperature || 0.7,
        top_p: params.top_p || 0.9,
        max_tokens: params.max_tokens || 256
      });
      
      return await fetchApi(`/test/model?${queryParams.toString()}`, {
        method: 'POST'
      });
    } catch (error) {
      console.error(`Error testing model:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Utility function to get active serving job
  const getActiveServingJob = useCallback(() => {
    return servingJobs.find(job => 
      job.status !== 'stopped' && job.status !== 'failed'
    );
  }, [servingJobs]);
  
  // Utility function to get active training job
  const getActiveTrainingJob = useCallback(() => {
    return trainingJobs.find(job => 
      job.status !== 'completed' && job.status !== 'failed' && job.status !== 'stopped'
    );
  }, [trainingJobs]);

  return {
    trainingJobs,
    servingJobs,
    loading,
    logsLoading,
    error,
    clearError,
    fetchJobs,
    startTrainingJob,
    stopTrainingJob,
    startServingJob,
    stopServingJob,
    fetchJobLogs,
    testModel,
    getActiveServingJob,
    getActiveTrainingJob
  };
};
