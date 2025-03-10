import { useState, useCallback } from 'react';
import { useApi } from './useApi';

/**
 * Hook for managing adapter data
 */
export const useAdapters = () => {
  const [adapters, setAdapters] = useState([]);
  const [loading, setLoading] = useState(false);
  const { fetchApi, error, clearError } = useApi();

  // Fetch all adapters
  const fetchAdapters = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchApi('/adapters');
      setAdapters(data);
      return data;
    } catch (error) {
      console.error("Error fetching adapters:", error);
      return [];
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Create a new adapter configuration
  const createAdapterConfig = useCallback(async (configData) => {
    setLoading(true);
    try {
      return await fetchApi('/configs/adapters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configData)
      });
    } catch (error) {
      console.error("Error creating adapter config:", error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Get a specific adapter configuration
  const getAdapterConfig = useCallback(async (name) => {
    setLoading(true);
    try {
      return await fetchApi(`/configs/adapters/${name}`);
    } catch (error) {
      console.error(`Error fetching adapter config ${name}:`, error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Delete an adapter
  const deleteAdapter = useCallback(async (name) => {
    setLoading(true);
    try {
      return await fetchApi(`/adapters/${name}`, {
        method: 'DELETE'
      });
    } catch (error) {
      console.error(`Error deleting adapter ${name}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Start training for an adapter
  const startTraining = useCallback(async (adapterName) => {
    setLoading(true);
    try {
      return await fetchApi(`/training/start?adapter_name=${adapterName}`, {
        method: 'POST'
      });
    } catch (error) {
      console.error(`Error starting training for ${adapterName}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Test an adapter
  const testAdapter = useCallback(async (adapterName, prompt, params = {}) => {
    setLoading(true);
    try {
      const queryParams = new URLSearchParams({
        adapter_name: adapterName,
        prompt: prompt,
        temperature: params.temperature || 0.7,
        top_p: params.top_p || 0.9,
        max_tokens: params.max_tokens || 256
      });
      
      return await fetchApi(`/test/adapter?${queryParams.toString()}`, {
        method: 'POST'
      });
    } catch (error) {
      console.error(`Error testing adapter ${adapterName}:`, error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  return {
    adapters,
    loading,
    error,
    clearError,
    fetchAdapters,
    createAdapterConfig,
    getAdapterConfig,
    deleteAdapter,
    startTraining,
    testAdapter
  };
};
