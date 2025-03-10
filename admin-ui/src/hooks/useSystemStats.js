import { useState, useCallback, useEffect, useRef } from 'react';
import { useApi } from './useApi';
import { POLLING_INTERVALS } from '../utils/constants';

/**
 * Hook for managing system statistics and metrics
 */
export const useSystemStats = (autoRefresh = false) => {
  const [systemStats, setSystemStats] = useState({
    gpu: { utilized: "0", temperature: "N/A", memory: "0/0 GB" },
    containers: [],
    disk_usage: {}
  });
  
  const [systemMetrics, setSystemMetrics] = useState({
    timestamps: [],
    gpu_utilization: [],
    memory_usage: []
  });
  
  const [loading, setLoading] = useState(false);
  const { fetchApi, error, clearError } = useApi();
  
  // Use refs to store interval IDs
  const intervalRef = useRef({});
  
  // Cleanup function for intervals
  const clearIntervals = useCallback(() => {
    Object.values(intervalRef.current).forEach(interval => {
      if (interval) clearInterval(interval);
    });
    intervalRef.current = {};
  }, []);

  // Fetch system stats
  const fetchSystemStats = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchApi('/system/stats');
      setSystemStats(data);
      return data;
    } catch (error) {
      console.error("Error fetching system stats:", error);
      return null;
    } finally {
      setLoading(false);
    }
  }, [fetchApi]);

  // Fetch system metrics (for charts)
  const fetchSystemMetrics = useCallback(async () => {
    try {
      const data = await fetchApi('/system/metrics');
      
      // Update metrics history
      setSystemMetrics(prev => {
        // Add new metrics data while maintaining limited history (30 points)
        const timestamps = [...prev.timestamps, data.timestamp].slice(-30);
        const gpuUtil = [...prev.gpu_utilization, parseFloat(data.gpu_utilization || 0)].slice(-30);
        const memUsage = [...prev.memory_usage, data.memory_usage?.percent_used || 0].slice(-30);
        
        return {
          timestamps,
          gpu_utilization: gpuUtil,
          memory_usage: memUsage
        };
      });
      
      return data;
    } catch (error) {
      console.error("Error fetching system metrics:", error);
      return null;
    }
  }, [fetchApi]);

  // Setup auto-refresh intervals
  useEffect(() => {
    if (autoRefresh) {
      // Setup intervals for auto-refresh
      intervalRef.current.stats = setInterval(fetchSystemStats, POLLING_INTERVALS.SYSTEM_STATS);
      intervalRef.current.metrics = setInterval(fetchSystemMetrics, POLLING_INTERVALS.SYSTEM_METRICS);
      
      // Initial fetch
      fetchSystemStats();
      fetchSystemMetrics();
    }
    
    // Cleanup on unmount
    return clearIntervals;
  }, [autoRefresh, fetchSystemStats, fetchSystemMetrics, clearIntervals]);

  return {
    systemStats,
    systemMetrics,
    loading,
    error,
    clearError,
    fetchSystemStats,
    fetchSystemMetrics,
    clearIntervals
  };
};
