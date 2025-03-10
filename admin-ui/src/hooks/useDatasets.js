import { useState, useCallback } from 'react';
import { useApi } from './useApi';

/**
 * Hook for managing dataset data
 */
export const useDatasets = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);
  const { fetchApi, error, clearError } = useApi();

  // Fetch all datasets
  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchApi('/datasets');
      setDatasets(data || []);
      return data;
    } catch (error) {
      console.error("Error fetching datasets:", error);
      return [];
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Upload a dataset
  const uploadDataset = useCallback(async (file) => {
    if (!file) return null;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      return await fetchApi('/datasets/upload', {
        method: 'POST',
        body: formData
      });
    } catch (error) {
      console.error("Error uploading dataset:", error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Delete a dataset
  const deleteDataset = useCallback(async (name) => {
    setLoading(true);
    try {
      return await fetchApi(`/datasets/${name}`, {
        method: 'DELETE'
      });
    } catch (error) {
      console.error(`Error deleting dataset ${name}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Preview a dataset (new function we'll implement)
  const previewDataset = useCallback(async (name, limit = 10) => {
    setLoading(true);
    try {
      return await fetchApi(`/datasets/${name}/preview?limit=${limit}`);
    } catch (error) {
      console.error(`Error previewing dataset ${name}:`, error);
      return { samples: [], error: error.message };
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  return {
    datasets,
    loading,
    error,
    clearError,
    fetchDatasets,
    uploadDataset,
    deleteDataset,
    previewDataset
  };
};
