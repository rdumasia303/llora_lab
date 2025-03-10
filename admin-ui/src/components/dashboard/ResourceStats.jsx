import React from 'react';
import { ChevronRight } from 'lucide-react';
import Card from '../common/Card';

/**
 * Resource statistics section for available models, adapters, etc.
 */
const ResourceStats = ({ 
  models, 
  adapters, 
  datasets, 
  jobs,
  containers,
  onNavigate 
}) => {
  return (
    <Card title="Available Resources" className="col-span-2 mt-6">
      <div className="grid grid-cols-4 gap-4 mb-4">
        <div className="bg-gray-700 p-3 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Models</span>
            <span className="text-lg font-semibold">{models.length}</span>
          </div>
          <button 
            onClick={() => onNavigate('models')}
            className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
          >
            View all <ChevronRight size={14} />
          </button>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Adapters</span>
            <span className="text-lg font-semibold">{adapters.length}</span>
          </div>
          <button 
            onClick={() => onNavigate('adapters')}
            className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
          >
            View all <ChevronRight size={14} />
          </button>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Datasets</span>
            <span className="text-lg font-semibold">{datasets.length}</span>
          </div>
          <button 
            onClick={() => onNavigate('datasets')}
            className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
          >
            View all <ChevronRight size={14} />
          </button>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Jobs</span>
            <span className="text-lg font-semibold">{jobs}</span>
          </div>
          <button 
            onClick={() => onNavigate('training')}
            className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
          >
            View all <ChevronRight size={14} />
          </button>
        </div>
      </div>
      
      {/* Container status */}
      <h4 className="text-sm font-medium mb-2">Docker Containers</h4>
      <div className="space-y-2">
        {containers.map(container => (
          <div key={container.id} className="flex justify-between text-xs bg-gray-750 p-2 rounded">
            <span className="text-gray-300">{container.name}</span>
            <span className={`${container.status === 'running' ? 'text-green-400' : 'text-yellow-400'}`}>
              {container.status}
            </span>
          </div>
        ))}
        {containers.length === 0 && (
          <div className="text-xs text-gray-400 p-2">
            No containers running
          </div>
        )}
      </div>
    </Card>
  );
};

export default ResourceStats;
