import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

/**
 * Error banner to display error messages
 */
const ErrorBanner = ({ message, onDismiss }) => {
  if (!message) return null;

  return (
    <div className="bg-red-800 text-white p-3 rounded mb-4 flex items-center animate-fadeIn">
      <AlertTriangle size={18} className="mr-2 flex-shrink-0" />
      <div className="flex-grow">{message}</div>
      {onDismiss && (
        <button 
          className="ml-auto text-white flex-shrink-0" 
          onClick={onDismiss}
          aria-label="Dismiss error"
        >
          <X size={18} />
        </button>
      )}
    </div>
  );
};

export default ErrorBanner;
