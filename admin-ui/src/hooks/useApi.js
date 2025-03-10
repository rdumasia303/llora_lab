import { useState, useCallback } from 'react';
import { API_URL } from '../utils/constants';

/**
 * Custom hook for API interactions
 */
export const useApi = () => {
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Generic fetch function with error handling
  const fetchApi = useCallback(async (endpoint, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}${endpoint}`, options);
      
      if (!response.ok) {
        let errorMessage = `API error (${response.status})`;
        try {
          const errorText = await response.text();
          errorMessage = `API error (${response.status}): ${errorText}`;
        } catch (e) {
          // If text extraction fails, use the original error
        }
        throw new Error(errorMessage);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      setError(`Error: ${error.message}`);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Clear any existing error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    fetchApi,
    error,
    clearError,
    isLoading
  };
};
