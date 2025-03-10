import React, { useState, useEffect, useCallback } from 'react';
import ModelList from './ModelList';
import ModelDialog from './ModelDialog';
import ConfirmDialog from '../common/ConfirmDialog';
import { Plus } from 'lucide-react';
import { useModels } from '../../hooks/useModels';

/**
 * Models management page
 */
const ModelsPage = ({ onSetActiveTab, setError, showConfirmation }) => {
  const { 
    models, 
    loading, 
    error, 
    fetchModels, 
    getModel, 
    createModel, 
    updateModel, 
    deleteModel 
  } = useModels();
  
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  
  // Initialize with data
  useEffect(() => {
    fetchModels().catch(err => {
      console.error("Error loading models:", err);
      setError("Failed to load models. Please try again.");
    });
  }, [fetchModels, setError]);
  
  // Set error from models hook to parent
  useEffect(() => {
    if (error) setError(error);
  }, [error, setError]);
  
  // Handle edit model
  const handleEditModel = useCallback(async (modelName) => {
    try {
      const modelData = await getModel(modelName);
      if (modelData) {
        setSelectedModel(modelData);
        setIsEditMode(true);
        setShowModelDialog(true);
      }
    } catch (err) {
      console.error("Error loading model for edit:", err);
      setError(`Failed to load model: ${err.message}`);
    }
  }, [getModel, setError]);
  
  // Handle save model
  const handleSaveModel = useCallback(async (modelData) => {
    try {
      if (isEditMode) {
        await updateModel(modelData.name, modelData);
      } else {
        await createModel(modelData);
      }
      
      setShowModelDialog(false);
      fetchModels();
    } catch (err) {
      console.error("Error saving model:", err);
      setError(`Failed to ${isEditMode ? 'update' : 'create'} model: ${err.message}`);
    }
  }, [isEditMode, updateModel, createModel, fetchModels, setError]);
  
  // Handle delete model
  const handleDeleteModel = useCallback((modelName) => {
    showConfirmation({
      title: "Delete Model Configuration",
      message: `Are you sure you want to delete the model configuration "${modelName}"?`,
      variant: "danger",
      onConfirm: async () => {
        try {
          await deleteModel(modelName);
          fetchModels();
        } catch (err) {
          console.error("Error deleting model:", err);
          setError(`Failed to delete model: ${err.message}`);
        }
      }
    });
  }, [deleteModel, fetchModels, setError, showConfirmation]);
  
  // Handle deploy model
  const handleDeployModel = useCallback((modelName) => {
    onSetActiveTab('serving');
  }, [onSetActiveTab]);
  
  // Handle add model
  const handleAddModel = useCallback(() => {
    setSelectedModel(null);
    setIsEditMode(false);
    setShowModelDialog(true);
  }, []);

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Models</h2>
        <button 
          onClick={handleAddModel}
          className="bg-green-700 hover:bg-green-600 text-white px-3 py-1 rounded text-sm flex items-center"
        >
          <Plus size={14} className="mr-1" /> Add Model
        </button>
      </div>
      
      <ModelList 
        models={models}
        loading={loading}
        onEdit={handleEditModel}
        onDelete={handleDeleteModel}
        onDeploy={handleDeployModel}
        onAddModel={handleAddModel}
      />
      
      <ModelDialog 
        isOpen={showModelDialog}
        onClose={() => setShowModelDialog(false)}
        model={selectedModel}
        isEditMode={isEditMode}
        onSave={handleSaveModel}
      />
    </div>
  );
};

export default ModelsPage;
