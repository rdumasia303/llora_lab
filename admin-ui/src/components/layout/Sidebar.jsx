import React from 'react';
import { 
  Monitor, 
  Database, 
  GitBranch, 
  Layers,
  RefreshCw, 
  Play, 
  Terminal
} from 'lucide-react';
import { TABS } from '../../utils/constants';

/**
 * Application sidebar with navigation
 */
const Sidebar = ({ activeTab, onTabChange }) => {
  // Navigation items with icons
  const navItems = [
    { id: TABS.OVERVIEW, label: 'Overview', icon: Monitor },
    { id: TABS.MODELS, label: 'Models', icon: Database },
    { id: TABS.ADAPTERS, label: 'Adapters', icon: GitBranch },
    { id: TABS.DATASETS, label: 'Datasets', icon: Layers },
    { id: TABS.TRAINING, label: 'Training', icon: RefreshCw },
    { id: TABS.SERVING, label: 'Serving', icon: Play },
    { id: TABS.LOGS, label: 'Logs', icon: Terminal }
  ];

  return (
    <aside className="w-48 bg-gray-800 border-r border-gray-700 p-4">
      <nav className="space-y-1">
        {navItems.map(item => (
          <button 
            key={item.id}
            onClick={() => onTabChange(item.id)}
            className={`flex items-center space-x-2 w-full text-left p-2 rounded-md transition-colors ${
              activeTab === item.id 
                ? 'bg-purple-800 text-white' 
                : 'hover:bg-gray-700'
            }`}
            aria-current={activeTab === item.id ? 'page' : undefined}
          >
            <item.icon size={18} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
};

export default Sidebar;
