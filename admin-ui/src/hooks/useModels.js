import { useState, useCallback, useEffect } from 'react';
import { useApi } from './useApi';

/**
 * Hook for managing model data
 */
export const useModels = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const { fetchApi, error, clearError } = useApi();

  // Fetch all models
  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchApi('/configs/models');
      setModels(data);
      return data;
    } catch (error) {
      console.error("Error fetching models:", error);
      return [];
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Get a specific model
  const getModel = useCallback(async (name) => {
    setLoading(true);
    try {
      return await fetchApi(`/configs/models/${name}`);
    } catch (error) {
      console.error(`Error fetching model ${name}:`, error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Create a new model
  const createModel = useCallback(async (modelData) => {
    setLoading(true);
    try {
      return await fetchApi('/configs/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelData)
      });
    } catch (error) {
      console.error("Error creating model:", error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Update an existing model
  const updateModel = useCallback(async (name, modelData) => {
    setLoading(true);
    try {
      return await fetchApi(`/configs/models/${name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelData)
      });
    } catch (error) {
      console.error(`Error updating model ${name}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Delete a model
  const deleteModel = useCallback(async (name) => {
    setLoading(true);
    try {
      return await fetchApi(`/configs/models/${name}`, {
        method: 'DELETE'
      });
    } catch (error) {
      console.error(`Error deleting model ${name}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  return {
    models,
    loading,
    error,
    clearError,
    fetchModels,
    getModel,
    createModel,
    updateModel,
    deleteModel
  };
};
