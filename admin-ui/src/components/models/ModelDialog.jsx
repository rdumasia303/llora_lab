import React from 'react';
import { X } from 'lucide-react';
import ModelForm from './ModelForm';

/**
 * Modal dialog for adding/editing models
 */
const ModelDialog = ({ 
  isOpen, 
  onClose, 
  model, 
  isEditMode = false, 
  onSave 
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">
            {isEditMode ? `Edit Model: ${model?.name}` : 'Add Model Configuration'}  
          </h3>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
          >
            <X size={24} />
          </button>
        </div>
        
        <ModelForm 
          model={model} 
          isEditMode={isEditMode} 
          onSave={onSave} 
          onCancel={onClose} 
        />
      </div>
    </div>
  );
};

export default ModelDialog;
