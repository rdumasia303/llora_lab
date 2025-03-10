import React, { useState, useEffect, useCallback } from 'react';
import JobSelector from './JobSelector';
import LogsViewer from './LogsViewer';
import { RefreshCw } from 'lucide-react';
import { useJobs } from '../../hooks/useJobs';

/**
 * Logs viewing page
 */
const LogsPage = ({ 
  selectedJobLogs, 
  setError 
}) => {
  const [logContent, setLogContent] = useState("");
  const [selectedJob, setSelectedJob] = useState(null);
  
  const {
    trainingJobs,
    servingJobs,
    loading,
    logsLoading,
    error,
    fetchJobs,
    fetchJobLogs
  } = useJobs();
  
  // Initialize with data
  useEffect(() => {
    fetchJobs().catch(err => {
      console.error("Error loading jobs:", err);
      setError("Failed to load jobs. Please try again.");
    });
    
    // Refresh jobs periodically
    const interval = setInterval(() => {
      fetchJobs().catch(console.error);
    }, 10000); // Less frequent updates for logs page
    
    return () => clearInterval(interval);
  }, [fetchJobs, setError]);
  
  // Set error from hooks to parent
  useEffect(() => {
    if (error) setError(error);
  }, [error, setError]);
  
  // Use selected job logs from props if available (when navigating from another page)
  useEffect(() => {
    if (selectedJobLogs && selectedJobLogs.id) {
      const jobType = selectedJobLogs.type;
      const jobs = jobType === 'training' ? trainingJobs : servingJobs;
      const job = jobs.find(j => j.id === selectedJobLogs.id);
      
      if (job) {
        handleSelectJob(selectedJobLogs.id, jobType, job);
      }
    }
  }, [selectedJobLogs, trainingJobs, servingJobs]);
  
  // Handle selecting a job
  const handleSelectJob = useCallback(async (jobId, jobType, job) => {
    setSelectedJob({
      id: jobId,
      type: jobType,
      job: job
    });
    
    // Clear log content before fetching new logs
    setLogContent("");
    
    // Fetch logs for the selected job
    try {
      const result = await fetchJobLogs(jobId, jobType);
      if (result && result.logs) {
        setLogContent(result.logs);
      }
    } catch (err) {
      console.error("Error fetching logs:", err);
      setLogContent(`Error fetching logs: ${err.message}`);
    }
  }, [fetchJobLogs]);
  
  // Handle refreshing logs
  const handleRefreshLogs = useCallback(async () => {
    if (!selectedJob) return;
    
    try {
      const result = await fetchJobLogs(selectedJob.id, selectedJob.type);
      if (result && result.logs) {
        setLogContent(result.logs);
      }
    } catch (err) {
      console.error("Error refreshing logs:", err);
    }
  }, [selectedJob, fetchJobLogs]);

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">System Logs</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleRefreshLogs}
            className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1 rounded text-sm flex items-center"
            disabled={!selectedJob || logsLoading}
          >
            <RefreshCw size={14} className={`mr-1 ${logsLoading ? 'animate-spin' : ''}`} />
            Refresh Logs
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar with job list */}
        <div className="lg:col-span-1">
          <JobSelector 
            trainingJobs={trainingJobs}
            servingJobs={servingJobs}
            selectedJobId={selectedJob?.id}
            onSelectJob={handleSelectJob}
            loading={loading}
          />
        </div>
        
        {/* Log content */}
        <div className="lg:col-span-3">
          <LogsViewer 
            logContent={logContent}
            loading={logsLoading}
            onRefresh={handleRefreshLogs}
            selectedJob={selectedJob}
          />
        </div>
      </div>
    </div>
  );
};

export default LogsPage;
