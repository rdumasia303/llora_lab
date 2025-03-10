import React from 'react';
import { 
  CheckCircle, 
  RefreshCw, 
  AlertTriangle, 
  Clock, 
  Pause
} from 'lucide-react';
import { STATUS_COLORS } from '../../utils/constants';

/**
 * Status badge that shows job status with appropriate styling
 */
const StatusBadge = ({ status, animate = true, showIcon = true }) => {
  // Get colors from constants
  const colors = STATUS_COLORS[status] || STATUS_COLORS.default;
  
  // Map status to icon and behavior
  let Icon;
  let animateIcon = animate;
  
  switch(status) {
    case 'completed':
      Icon = CheckCircle;
      animateIcon = false;
      break;
    case 'failed':
      Icon = AlertTriangle;
      animateIcon = false;
      break;
    case 'stopped':
      Icon = Pause;
      animateIcon = false;
      break;
    case 'running':
    case 'initializing':
    case 'starting':
      Icon = RefreshCw;
      break;
    case 'ready':
      Icon = CheckCircle;
      animateIcon = false;
      break;
    default:
      Icon = Clock;
      animateIcon = false;
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors.bg} ${colors.text}`}>
      {showIcon && (
        <Icon 
          size={12} 
          className={`mr-1 ${animateIcon ? 'animate-spin' : ''}`} 
        />
      )}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

export default StatusBadge;
