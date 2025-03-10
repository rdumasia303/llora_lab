import React from 'react';

/**
 * Reusable card component for containing content
 */
const Card = ({ 
  children, 
  title, 
  className = '', 
  bodyClassName = '', 
  headerClassName = '',
  headerAction,
  footer
}) => {
  return (
    <div className={`bg-gray-800 rounded-lg border border-gray-700 overflow-hidden ${className}`}>
      {title && (
        <div className={`border-b border-gray-700 p-4 flex justify-between items-center ${headerClassName}`}>
          <h3 className="text-lg font-semibold">{title}</h3>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      <div className={`p-4 ${bodyClassName}`}>
        {children}
      </div>
      {footer && (
        <div className="border-t border-gray-700 p-4">
          {footer}
        </div>
      )}
    </div>
  );
};

export default Card;
