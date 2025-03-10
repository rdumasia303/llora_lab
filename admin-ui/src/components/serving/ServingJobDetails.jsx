import React from 'react';
import Card from '../common/Card';
import StatusBadge from '../common/StatusBadge';
import { formatTimestamp } from '../../utils/formatters';
import { CheckCircle, RefreshCw, AlertTriangle, ExternalLink, Terminal } from 'lucide-react';

/**
 * Displays details for the active serving job
 */
const ServingJobDetails = ({ 
  job, 
  onViewLogs, 
  onStopServing 
}) => {
  if (!job) return null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card className="lg:col-span-2">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-xl font-semibold mb-1">Active Model: {job.model_conf}</h3>
            {job.adapter && (
              <div className="text-sm text-purple-400 mb-2">
                with adapter: {job.adapter}
              </div>
            )}
            <div className="flex items-center mb-4">
              <StatusBadge status={job.status} />
            </div>
          </div>
          <button 
            onClick={onStopServing}
            className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
            Stop Serving
          </button>
        </div>
        
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-750 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-400 mb-2">Status</h4>
            <p className="text-lg font-medium">
              {job.status === 'ready' ? 'Online' : job.status}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Started: {formatTimestamp(job.start_time)}
            </p>
          </div>
          
          <div className="bg-gray-750 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-400 mb-2">Requests</h4>
            <p className="text-lg font-medium">{job.requests_served}</p>
            <p className="text-xs text-gray-500 mt-1">
              Avg. response time: {job.avg_response_time.toFixed(2)}ms
            </p>
          </div>
        </div>
        
        {job.message && (
          <div className="bg-gray-750 p-3 rounded-lg mb-4 text-sm">
            <h4 className="text-xs font-medium text-gray-400 mb-1">Message</h4>
            <p className="text-gray-300">{job.message}</p>
          </div>
        )}
        
        {job.status === 'ready' && (
          <div className="bg-gray-750 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-400 mb-2">API Endpoints</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">OpenAI API:</span>
                <code className="bg-gray-800 px-2 py-1 rounded text-xs">http://localhost:8000/v1</code>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Web UI:</span>
                <a 
                  href="http://localhost:3000" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center text-purple-400 hover:text-purple-300 text-xs"
                >
                  http://localhost:3000 <ExternalLink size={12} className="ml-1" />
                </a>
              </div>
            </div>
          </div>
        )}
        
        <div className="mt-4">
          <button
            onClick={onViewLogs}
            className="text-purple-400 hover:text-purple-300 text-sm flex items-center"
          >
            <Terminal size={14} className="mr-1" />
            View Logs
          </button>
        </div>
      </Card>
      
      <Card title="Quick Test">
        <p className="text-sm text-gray-400 mb-4">
          You can quickly test your deployed model through the form below or by accessing the Web UI for a more complete interface.
        </p>
        
        <div className="py-4 border-t border-gray-700">
          <a 
            href="http://localhost:3000" 
            target="_blank" 
            rel="noopener noreferrer"
            className="bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded-md flex items-center justify-center"
          >
            <ExternalLink size={16} className="mr-2" />
            Open Web Interface
          </a>
        </div>
      </Card>
    </div>
  );
};

export default ServingJobDetails;
