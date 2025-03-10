import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer, 
  AreaChart, 
  Area 
} from 'recharts';
import Card from '../common/Card';
import LoadingSpinner from '../common/LoadingSpinner';
import { AlertTriangle } from 'lucide-react';

/**
 * System metrics charts
 */
const SystemMetrics = ({ systemMetricsData, trainingChartData, loading }) => {
  // Format timestamp for X-axis
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
  };

  return (
    <div className="grid grid-cols-2 gap-4 mt-6">
      {/* System Metrics Chart */}
      <Card title="System Metrics">
        <div className="h-64">
          {loading && systemMetricsData.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <LoadingSpinner message="Collecting metrics data..." />
            </div>
          ) : systemMetricsData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={systemMetricsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" />
                <XAxis 
                  dataKey="timestamp"
                  tickFormatter={formatTimestamp}
                  stroke="#9ca3af" 
                />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{backgroundColor: '#374151', borderColor: '#4b5563'}}
                  labelStyle={{color: '#e5e7eb'}}
                  formatter={(value, name) => {
                    return [`${value.toFixed(2)}%`, name === 'gpu' ? 'GPU Usage' : 'Memory Usage'];
                  }}
                  labelFormatter={(timestamp) => {
                    return new Date(timestamp).toLocaleTimeString();
                  }}
                />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="gpu" 
                  name="GPU Utilization" 
                  stroke="#8884d8" 
                  fill="#8884d8" 
                  fillOpacity={0.3} 
                />
                <Area 
                  type="monotone" 
                  dataKey="memory" 
                  name="Memory Usage" 
                  stroke="#82ca9d" 
                  fill="#82ca9d" 
                  fillOpacity={0.3} 
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400">
              <div className="text-center">
                <AlertTriangle size={36} className="mx-auto mb-2" />
                <p>No metrics data available</p>
              </div>
            </div>
          )}
        </div>
      </Card>
      
      {/* Training Progress Chart */}
      <Card title="Training Progress">
        <div className="h-64">
          {trainingChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a4a4a" />
                <XAxis dataKey="step" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip 
                  contentStyle={{backgroundColor: '#374151', borderColor: '#4b5563'}}
                  labelStyle={{color: '#e5e7eb'}}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }} 
                  isAnimationActive={false}
                />
                {trainingChartData[0]?.learningRate && (
                  <Line 
                    type="monotone" 
                    dataKey="learningRate" 
                    stroke="#82ca9d" 
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-400">
              <div className="text-center">
                <AlertTriangle size={36} className="mx-auto mb-2" />
                <p>No training data available</p>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default SystemMetrics;
