import React, { useState } from 'react';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import { AlertTriangle, Play, RefreshCw, Info } from 'lucide-react';

/**
 * Interface for testing adapters (currently shown as "Coming Soon")
 */
const AdapterTestInterface = ({ 
  adapters, 
  selectedAdapter, 
  onSelectAdapter, 
  onNavigateToServing 
}) => {
  // NOTE: This is replaced with "Coming Soon" message as per requirements
  
  return (
    <Card title="Test Adapter">
      <div className="flex flex-col items-center justify-center text-center py-12">
        <AlertTriangle size={48} className="text-yellow-500 mb-4" />
        <h4 className="text-xl font-semibold mb-2">Coming Soon</h4>
        <p className="text-gray-400 max-w-md">
          Direct adapter testing is currently being improved. For now, please deploy your adapter 
          with its base model to test it via the serving interface.
        </p>
        <button 
          onClick={onNavigateToServing}
          className="mt-6 bg-purple-700 hover:bg-purple-600 text-white px-4 py-2 rounded-md"
        >
          Go to Model Serving
        </button>
      </div>
    </Card>
  );
};

export default AdapterTestInterface;
