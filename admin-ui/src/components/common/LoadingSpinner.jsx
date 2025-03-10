import React from 'react';
import { RefreshCw } from 'lucide-react';

/**
 * Loading spinner with optional text
 */
const LoadingSpinner = ({ 
  size = 'medium', 
  message = 'Loading...', 
  fullScreen = false,
  className = '' 
}) => {
  // Size mapping
  const sizeMap = {
    small: { icon: 16, text: 'text-sm' },
    medium: { icon: 24, text: 'text-base' },
    large: { icon: 36, text: 'text-lg' }
  };
  
  const { icon, text } = sizeMap[size] || sizeMap.medium;
  
  const spinner = (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <RefreshCw 
        size={icon} 
        className="animate-spin text-purple-500 mb-2" 
      />
      {message && <p className={`${text} text-gray-300`}>{message}</p>}
    </div>
  );
  
  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-gray-900 bg-opacity-70 flex items-center justify-center z-50">
        {spinner}
      </div>
    );
  }
  
  return spinner;
};

export default LoadingSpinner;
