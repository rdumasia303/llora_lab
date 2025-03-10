import React from 'react';
import { Coffee, Book, Settings } from 'lucide-react';

/**
 * Application header
 */
const Header = () => {
  return (
    <header className="bg-gray-800 p-4 border-b border-gray-700 flex justify-between items-center">
      <div className="flex items-center space-x-2">
        <Coffee className="h-8 w-8 text-purple-400" />
        <h1 className="text-xl font-bold">Llora Lab</h1>
      </div>
      <div className="flex items-center space-x-4">
        <a 
          href="https://github.com/your-repo/llora-lab/docs" 
          target="_blank" 
          rel="noopener noreferrer"
          className="bg-gray-700 px-3 py-1 rounded-md hover:bg-gray-600 transition-colors text-sm flex items-center"
        >
          <Book size={14} className="mr-1" />
          Documentation
        </a>
        <button className="bg-gray-700 px-3 py-1 rounded-md hover:bg-gray-600 transition-colors text-sm flex items-center">
          <Settings size={14} className="mr-1" />
          Settings
        </button>
      </div>
    </header>
  );
};

export default Header;
