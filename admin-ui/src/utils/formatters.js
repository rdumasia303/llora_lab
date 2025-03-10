/**
 * Utility functions for formatting data
 */

// Format file sizes in human-readable form
export const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  // Format a timestamp to a readable date
  export const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    try {
      return new Date(timestamp).toLocaleString();
    } catch (e) {
      return timestamp;
    }
  };
  
  // Format a number with commas for thousands
  export const formatNumber = (num) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };
  
  // Format a duration in seconds to mm:ss format
  export const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Format a percentage
  export const formatPercent = (value, precision = 1) => {
    if (value === null || value === undefined) return 'N/A';
    return `${value.toFixed(precision)}%`;
  };
  
  // Format a floating point number
  export const formatFloat = (value, precision = 4) => {
    if (value === null || value === undefined) return 'N/A';
    return value.toFixed(precision);
  };
  
  // Truncate a string with ellipsis
  export const truncateString = (str, maxLength = 50) => {
    if (!str) return '';
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
  };
  