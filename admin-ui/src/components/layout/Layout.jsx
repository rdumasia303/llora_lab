import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import ErrorBanner from '../common/ErrorBanner';

/**
 * Main application layout
 */
const Layout = ({ 
  children, 
  activeTab, 
  onTabChange, 
  error, 
  onClearError 
}) => {
  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-200">
      <Header />
      
      <div className="flex flex-1">
        <Sidebar activeTab={activeTab} onTabChange={onTabChange} />
        
        <main className="flex-1 p-6 overflow-auto">
          {error && (
            <ErrorBanner 
              message={error} 
              onDismiss={onClearError} 
            />
          )}
          
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
