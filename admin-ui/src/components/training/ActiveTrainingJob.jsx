import React, { useState, useEffect, useRef } from 'react';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import StatusBadge from '../common/StatusBadge';
import { RefreshCw, Terminal } from 'lucide-react';
import { formatPercent, formatFloat } from '../../utils/formatters';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

/**
 * Displays details for the active training job with loss graph
 */
const ActiveTrainingJob = ({ 
  job, 
  onViewLogs, 
  onStopTraining 
}) => {
  const [lossHistory, setLossHistory] = useState([]);
  const previousLossRef = useRef(null);
  
  // Track loss history for the chart
  useEffect(() => {
    if (!job || job.loss === null || job.loss === undefined) return;
    
    // Only add new data points when loss changes
    if (previousLossRef.current !== job.loss) {
      setLossHistory(prev => {
        // Keep only the last 50 data points to avoid overcrowding
        const newHistory = [
          ...prev,
          {
            step: job.step,
            loss: job.loss
          }
        ];
        
        // Keep the chart showing the most recent 50 points
        if (newHistory.length > 50) {
          return newHistory.slice(newHistory.length - 50);
        }
        return newHistory;
      });
      
      previousLossRef.current = job.loss;
    }
  }, [job?.loss, job?.step]);

  if (!job) return null;

  const progress = (job.step / job.total_steps) * 100;

  return (
    <Card className="mb-6">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h3 className="text-xl font-semibold flex items-center">
            <RefreshCw className="h-5 w-5 mr-2 text-yellow-400 animate-spin" />
            Active Training: {job.adapter_config}
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Started: {new Date(job.start_time).toLocaleString()}
          </p>
        </div>
        
        <button 
          onClick={() => onStopTraining(job.id)}
          className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1"><path d="M18 6L6 18"></path><path d="M6 6l12 12"></path></svg>
          Stop Training
        </button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span>Progress</span>
              <span>
                {job.step} / {job.total_steps} steps ({Math.round(progress)}%)
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2.5">
              <div 
                className="bg-yellow-600 h-2.5 rounded-full" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-750 p-3 rounded-lg">
              <h4 className="text-xs uppercase text-gray-400 mb-1">Current Loss</h4>
              <p className="text-xl font-semibold">
                {job.loss !== null && job.loss !== undefined 
                  ? formatFloat(job.loss, 4) 
                  : 'N/A'}
              </p>
            </div>
            
            <div className="bg-gray-750 p-3 rounded-lg">
              <h4 className="text-xs uppercase text-gray-400 mb-1">Learning Rate</h4>
              <p className="text-xl font-semibold">
                {job.learning_rate !== null && job.learning_rate !== undefined 
                  ? job.learning_rate.toExponential(2) 
                  : 'N/A'}
              </p>
            </div>
          </div>
          
          {job.message && (
            <div className="mt-4 bg-gray-750 p-3 rounded-lg">
              <h4 className="text-xs uppercase text-gray-400 mb-1">Status Message</h4>
              <p className="text-sm">{job.message}</p>
            </div>
          )}
          
          <div className="mt-4">
            <button
              onClick={() => onViewLogs(job)}
              className="text-purple-400 hover:text-purple-300 text-sm flex items-center"
            >
              <Terminal size={14} className="mr-1" />
              View Full Logs
            </button>
          </div>
        </div>
        
        <div className="bg-gray-750 rounded-lg p-4">
          {job.step > 0 && lossHistory.length > 1 ? (
            <div className="w-full h-64">
              <div className="mb-2 text-sm font-medium text-gray-400">Training Loss</div>
              <ResponsiveContainer width="100%" height="90%">
                <LineChart
                  data={lossHistory}
                  margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#555" />
                  <XAxis 
                    dataKey="step" 
                    label={{ value: 'Step', position: 'insideBottomRight', offset: -5 }}
                    stroke="#999"
                    tickLine={{ stroke: '#999' }}
                  />
                  <YAxis 
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                    stroke="#999"
                    tickLine={{ stroke: '#999' }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#333', border: '1px solid #555' }}
                    formatter={(value) => [formatFloat(value, 5), 'Loss']}
                    labelFormatter={(step) => `Step: ${step}`}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#EAB308" 
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="w-full h-64 flex items-center justify-center text-center text-gray-400">
              {job.step === 0 ? (
                <div>
                  <p>Initializing training job...</p>
                  <LoadingSpinner className="mt-2" />
                </div>
              ) : (
                <div>
                  <p>Waiting for training metrics...</p>
                  <LoadingSpinner className="mt-2" />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default ActiveTrainingJob;