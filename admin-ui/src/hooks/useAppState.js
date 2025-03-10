import { useState, useCallback, useMemo } from 'react';
import { TABS } from '../utils/constants';

/**
 * Hook for managing application state to reduce re-renders
 */
export const useAppState = () => {
  const [activeTab, setActiveTab] = useState(TABS.OVERVIEW);
  const [error, setError] = useState(null);
  
  // Sub-tab states
  const [modelsTab, setModelsTab] = useState('list');
  const [adaptersTab, setAdaptersTab] = useState('list');
  const [datasetsTab, setDatasetsTab] = useState('list');
  const [trainingTab, setTrainingTab] = useState('list');
  
  // Dialog states
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [showAdapterDialog, setShowAdapterDialog] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [confirmDialogProps, setConfirmDialogProps] = useState({
    message: '',
    onConfirm: () => {},
    title: 'Are you sure?',
    variant: 'warning'
  });
  
  // Selected item states
  const [selectedJobLogs, setSelectedJobLogs] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState(null);
  
  // Clear error message
  const clearError = useCallback(() => setError(null), []);
  
  // Confirmation dialog helpers
  const showConfirmation = useCallback((props) => {
    setConfirmDialogProps({
      message: '',
      onConfirm: () => {},
      title: 'Are you sure?',
      variant: 'warning',
      ...props
    });
    setShowConfirmDialog(true);
  }, []);
  
  const closeConfirmDialog = useCallback(() => {
    setShowConfirmDialog(false);
  }, []);
  
  // Memoized compound states to optimize renders
  const tabStates = useMemo(() => ({
    activeTab,
    modelsTab,
    adaptersTab, 
    datasetsTab,
    trainingTab
  }), [activeTab, modelsTab, adaptersTab, datasetsTab, trainingTab]);
  
  const dialogStates = useMemo(() => ({
    showModelDialog,
    showAdapterDialog,
    showConfirmDialog,
    confirmDialogProps
  }), [showModelDialog, showAdapterDialog, showConfirmDialog, confirmDialogProps]);
  
  const selectionStates = useMemo(() => ({
    selectedJobLogs,
    selectedDataset
  }), [selectedJobLogs, selectedDataset]);

  return {
    // Tab states
    activeTab,
    setActiveTab,
    modelsTab,
    setModelsTab,
    adaptersTab,
    setAdaptersTab,
    datasetsTab,
    setDatasetsTab,
    trainingTab,
    setTrainingTab,
    tabStates,
    
    // Error handling
    error,
    setError,
    clearError,
    
    // Dialog states
    showModelDialog,
    setShowModelDialog,
    showAdapterDialog,
    setShowAdapterDialog,
    showConfirmDialog,
    setShowConfirmDialog,
    confirmDialogProps,
    showConfirmation,
    closeConfirmDialog,
    dialogStates,
    
    // Selection states
    selectedJobLogs,
    setSelectedJobLogs,
    selectedDataset,
    setSelectedDataset,
    selectionStates
  };
};
