import React from 'react';
import { RefreshCw, Server } from 'lucide-react';
import StatusBadge from '../common/StatusBadge';
import { formatTimestamp } from '../../utils/formatters';
import LoadingSpinner from '../common/LoadingSpinner';

/**
 * Sidebar with job list for selecting logs to view
 */
const JobSelector = ({
  trainingJobs,
  servingJobs,
  selectedJobId,
  onSelectJob,
  loading
}) => {
  // Sort jobs by start time, most recent first
  const sortedTrainingJobs = [...trainingJobs].sort((a, b) => 
    new Date(b.start_time) - new Date(a.start_time)
  );
  
  const sortedServingJobs = [...servingJobs].sort((a, b) => 
    new Date(b.start_time) - new Date(a.start_time)
  );

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 h-[calc(100vh-200px)] overflow-y-auto">
      <h3 className="text-lg font-semibold mb-4">Jobs</h3>
      
      {loading ? (
        <div className="text-center py-12">
          <LoadingSpinner message="Loading jobs..." />
        </div>
      ) : (
        <>
          {/* Training jobs */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center">
              <RefreshCw size={14} className="mr-1" /> Training Jobs
            </h4>
            
            <div className="space-y-2">
              {sortedTrainingJobs.length === 0 ? (
                <p className="text-sm text-gray-500">No training jobs</p>
              ) : (
                sortedTrainingJobs.map(job => (
                  <button
                    key={job.id}
                    onClick={() => onSelectJob(job.id, 'training', job)}
                    className={`w-full text-left p-2 rounded-md text-sm ${
                      selectedJobId === job.id 
                        ? 'bg-purple-800 text-white' 
                        : 'hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="truncate">{job.adapter_config}</span>
                      <StatusBadge status={job.status} showIcon={false} />
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      {formatTimestamp(job.start_time)}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
          
          {/* Serving jobs */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 mb-2 flex items-center">
              <Server size={14} className="mr-1" /> Serving Jobs
            </h4>
            
            <div className="space-y-2">
              {sortedServingJobs.length === 0 ? (
                <p className="text-sm text-gray-500">No serving jobs</p>
              ) : (
                sortedServingJobs.map(job => (
                  <button
                    key={job.id}
                    onClick={() => onSelectJob(job.id, 'serving', job)}
                    className={`w-full text-left p-2 rounded-md text-sm ${
                      selectedJobId === job.id 
                        ? 'bg-purple-800 text-white' 
                        : 'hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="truncate">{job.model_conf}</span>
                      <StatusBadge status={job.status} showIcon={false} />
                    </div>
                    {job.adapter && (
                      <div className="text-xs text-purple-400">
                        with adapter: {job.adapter}
                      </div>
                    )}
                    <div className="text-xs text-gray-400 mt-1">
                      {formatTimestamp(job.start_time)}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default JobSelector;
