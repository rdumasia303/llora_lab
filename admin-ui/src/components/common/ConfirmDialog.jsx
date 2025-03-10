import React from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * Reusable confirmation dialog
 */
const ConfirmDialog = ({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title = "Are you sure?", 
  message, 
  confirmText = "Confirm", 
  cancelText = "Cancel",
  variant = "warning" // warning, danger, info
}) => {
  if (!isOpen) return null;
  
  // Determine icon and colors based on variant
  let Icon = AlertTriangle;
  let iconColor = "text-yellow-500";
  let confirmButtonClass = "bg-red-700 hover:bg-red-600";
  
  if (variant === "danger") {
    iconColor = "text-red-500";
    confirmButtonClass = "bg-red-700 hover:bg-red-600";
  } else if (variant === "info") {
    Icon = () => <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>;
    iconColor = "text-blue-500";
    confirmButtonClass = "bg-blue-700 hover:bg-blue-600";
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full">
        <div className="flex flex-col items-center mb-4">
          <Icon size={48} className={iconColor + " mb-2"} />
          <h3 className="text-lg font-semibold text-center">{title}</h3>
          {message && <p className="text-center mt-2 text-gray-300">{message}</p>}
        </div>
        
        <div className="flex justify-center space-x-4">
          <button 
            onClick={onClose}
            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
          >
            {cancelText}
          </button>
          <button 
            onClick={() => {
              onConfirm();
              onClose();
            }}
            className={`${confirmButtonClass} text-white px-4 py-2 rounded-md`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;
