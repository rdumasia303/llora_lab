import React from 'react';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';
import { formatTimestamp } from '../../utils/formatters';

/**
 * List of training jobs
 */
const TrainingJobList = ({ 
  jobs, 
  onViewLogs, 
  onStopTraining, 
  onDeploy 
}) => {
  return (
    <Card title="Training History" className="overflow-hidden">
      <table className="min-w-full divide-y divide-gray-700">
        <thead className="bg-gray-750">
          <tr>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Adapter</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Progress</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Start Time</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-700">
          {jobs.length === 0 ? (
            <tr>
              <td colSpan="5" className="px-6 py-8 text-center text-gray-400">
                No training jobs found
              </td>
            </tr>
          ) : (
            jobs.map(job => (
              <tr 
                key={job.id} 
                className={
                  job.status !== 'completed' && 
                  job.status !== 'failed' && 
                  job.status !== 'stopped' 
                    ? 'bg-yellow-900 bg-opacity-20' 
                    : ''
                }
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium">{job.adapter_config}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <StatusBadge status={job.status} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  {job.status === 'completed' ? (
                    '100%'
                  ) : (
                    <div>
                      <div className="text-xs mb-1">{job.step} / {job.total_steps} steps</div>
                      <div className="w-32 bg-gray-700 rounded-full h-1.5">
                        <div 
                          className={`h-1.5 rounded-full ${
                            job.status === 'failed' || job.status === 'stopped' 
                              ? 'bg-red-600' 
                              : 'bg-yellow-600'
                          }`}
                          style={{
                            width: `${(job.step / job.total_steps) * 100}%`
                          }}
                        ></div>
                      </div>
                    </div>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  {formatTimestamp(job.start_time)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="flex space-x-3">
                    <button
                      onClick={() => onViewLogs(job)}
                      className="text-blue-400 hover:text-blue-300"
                    >
                      Logs
                    </button>
                    
                    {job.status !== 'completed' && 
                     job.status !== 'failed' && 
                     job.status !== 'stopped' && (
                      <button
                        onClick={() => onStopTraining(job.id)}
                        className="text-red-400 hover:text-red-300"
                      >
                        Stop
                      </button>
                    )}
                    
                    {job.status === 'completed' && (
                      <button
                        onClick={() => onDeploy(job)}
                        className="text-green-400 hover:text-green-300"
                      >
                        Deploy
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </Card>
  );
};

export default TrainingJobList;
