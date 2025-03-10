import React from 'react';
import { CheckCircle, RefreshCw, AlertTriangle, ChevronRight } from 'lucide-react';
import Card from '../common/Card';
import { formatPercent } from '../../utils/formatters';

/**
 * System overview cards showing status and resources
 */
const SystemOverview = ({ 
  systemStats, 
  activeServingJob, 
  activeTrainingJob, 
  onNavigate,
  onStopServing,
  onStopTraining
}) => {
  return (
    <div className="grid grid-cols-3 gap-4">
      {/* System Status Card */}
      <Card>
        <h3 className="text-lg font-semibold mb-2">System Status</h3>
        <div className="flex items-center space-x-2 text-green-400">
          <CheckCircle size={18} />
          <span>All systems operational</span>
        </div>
        <div className="mt-4">
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
      </Card>
      
      {/* Active Model Card */}
      <Card>
        <h3 className="text-lg font-semibold mb-2">Active Model</h3>
        {activeServingJob ? (
          <>
            <p className="text-xl">{activeServingJob.model_conf}</p>
            {activeServingJob.adapter && <p className="text-sm text-gray-400">with {activeServingJob.adapter}</p>}
            <div className="mt-4 flex justify-between items-center">
              <span className="text-sm text-gray-400">Status: {activeServingJob.status}</span>
              <button 
                onClick={() => onStopServing(activeServingJob.id)}
                className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm"
              >
                Stop
              </button>
            </div>
          </>
        ) : (
          <>
            <p className="text-gray-400">No model currently active</p>
            <div className="mt-4">
              <button 
                onClick={() => onNavigate('serving')}
                className="bg-purple-700 hover:bg-purple-600 text-white px-3 py-1 rounded text-sm"
              >
                Deploy Model
              </button>
            </div>
          </>
        )}
      </Card>
      
      {/* Training Status Card */}
      <Card>
        <h3 className="text-lg font-semibold mb-2">Training Status</h3>
        {activeTrainingJob ? (
          <>
            <div className="flex items-center space-x-2 text-yellow-400">
              <RefreshCw size={18} />
              <span>Training in progress</span>
            </div>
            <div className="mt-4">
              <p className="text-sm">{activeTrainingJob.adapter_config} adapter</p>
              <p className="text-xs text-gray-400">
                Step {activeTrainingJob.step}/{activeTrainingJob.total_steps} • 
                Loss: {activeTrainingJob.loss?.toFixed(2) || 'N/A'}
              </p>
              <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
                <div 
                  className="bg-yellow-600 h-2.5 rounded-full" 
                  style={{
                    width: `${(activeTrainingJob.step / activeTrainingJob.total_steps) * 100}%`
                  }}
                ></div>
              </div>
              <div className="mt-2">
                <button 
                  onClick={() => onStopTraining(activeTrainingJob.id)}
                  className="bg-red-700 hover:bg-red-600 text-white px-3 py-1 rounded text-sm"
                >
                  Stop Training
                </button>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center space-x-2 text-gray-400">
              <CheckCircle size={18} />
              <span>No active training jobs</span>
            </div>
            <div className="mt-4">
              <button 
                onClick={() => onNavigate('training')}
                className="bg-purple-700 hover:bg-purple-600 text-white px-3 py-1 rounded text-sm"
              >
                Start Training
              </button>
            </div>
          </>
        )}
      </Card>
    </div>
  );
};

export default SystemOverview;
