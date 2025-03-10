import React, { useEffect, useRef } from 'react';
import { RefreshCw, ChevronRight, Terminal, Info } from 'lucide-react';

/**
 * Component for viewing job logs
 */
const LogsViewer = ({ 
  logContent, 
  loading, 
  onRefresh, 
  selectedJob 
}) => {
  const contentRef = useRef(null);
  
  // Scroll to bottom when new logs come in
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [logContent]);

  // Scroll functions
  const scrollToTop = () => {
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  };
  
  const scrollToBottom = () => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  };
  
  // Copy logs to clipboard
  const copyLogs = () => {
    if (logContent) {
      navigator.clipboard.writeText(logContent);
      alert('Logs copied to clipboard');
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 h-[calc(100vh-200px)] flex flex-col">
      <div className="bg-gray-750 border-b border-gray-700 p-3 flex justify-between items-center">
        <div>
          <h3 className="font-medium">
            {selectedJob ? (
              <>
                Logs: {selectedJob.type === 'training' ? 
                  selectedJob.job?.adapter_config :
                  selectedJob.job?.model_conf
                }
              </>
            ) : (
              'Select a job to view logs'
            )}
          </h3>
          {selectedJob && selectedJob.job && (
            <div className="text-xs text-gray-400">
              {selectedJob.type === 'training' ? 
                `Training Job` :
                `Serving Job`
              }
              {' • '}
              {selectedJob.job.status}
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button 
            className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
            onClick={scrollToTop}
            title="Scroll to top"
          >
            <ChevronRight className="h-4 w-4 transform rotate-90" />
          </button>
          <button 
            className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
            onClick={scrollToBottom}
            title="Scroll to bottom"
          >
            <ChevronRight className="h-4 w-4 transform -rotate-90" />
          </button>
          <button 
            className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
            onClick={copyLogs}
            title="Copy to clipboard"
            disabled={!logContent}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
          <button 
            className="text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 p-1 rounded"
            onClick={onRefresh}
            title="Refresh logs"
            disabled={loading || !selectedJob}
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>
      
      <div 
        ref={contentRef}
        className="flex-1 p-4 bg-gray-900 font-mono text-xs overflow-y-auto"
      >
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-purple-500" />
              <p>Loading logs...</p>
            </div>
          </div>
        ) : !selectedJob ? (
          <div className="h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <Terminal size={32} className="mx-auto mb-2" />
              <p>Select a job from the sidebar to view logs</p>
            </div>
          </div>
        ) : logContent ? (
          logContent.split('\n').map((line, i) => (
            <div key={i} className="text-gray-300 whitespace-pre-wrap">{line || '\u00A0'}</div>
          ))
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <Info size={32} className="mx-auto mb-2" />
              <p>No logs available for this job</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LogsViewer;
