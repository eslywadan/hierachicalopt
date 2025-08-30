// src/components/TFTLCDDashboard.js
import React, { useState } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { Calendar, TrendingUp, DollarSign, Package, AlertTriangle, BarChart3, Activity, RefreshCw } from 'lucide-react';
import useTFTLCDData from '../hooks/useTFTLCDData';
import './TFTLCDDashboard.css';

const TFTLCDDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedMetric, setSelectedMetric] = useState('revenue');

  // Use the real data hook
  const {
    weeklyTrends,
    plantPerformance,
    sizeAnalysis,
    supplyChainData,
    qualityMetrics,
    kpis,
    dataSummary,
    filters,
    loading,
    error,
    updateFilter,
    resetFilters,
    refreshData
  } = useTFTLCDData();

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1'];

  const MetricCard = ({ title, value, unit, icon: Icon, color }) => (
    <div className="metric-card">
      <div className="metric-card-content">
        <div className="metric-card-text">
          <p className="metric-card-title">{title}</p>
          <p className="metric-card-value">{value}{unit}</p>
        </div>
        <div className={`metric-card-icon ${color}`}>
          <Icon className="icon" />
        </div>
      </div>
    </div>
  );

  const TabButton = ({ id, label, icon: Icon }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`tab-button ${activeTab === id ? 'tab-button-active' : ''}`}
    >
      <Icon className="tab-icon" />
      {label}
    </button>
  );

  const LoadingSpinner = () => (
    <div className="loading-container">
      <RefreshCw className="loading-spinner" />
      <p>Loading TFT-LCD data...</p>
    </div>
  );

  const ErrorMessage = ({ error, onRetry }) => (
    <div className="error-container">
      <AlertTriangle className="error-icon" />
      <h3>Error Loading Data</h3>
      <p>{error}</p>
      <button onClick={onRetry} className="error-retry-button">
        <RefreshCw className="button-icon" />
        Retry
      </button>
    </div>
  );

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage error={error} onRetry={refreshData} />;
  }

  if (!dataSummary) {
    return (
      <div className="no-data-container">
        <Package className="no-data-icon" />
        <h3>No Data Available</h3>
        <p>Please ensure CSV files are available in the data folder.</p>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">TFT-LCD Manufacturing Analytics Dashboard</h1>
        <p className="dashboard-subtitle">
          Real-time analysis of {dataSummary.mainRecords?.toLocaleString()} production records 
          from {dataSummary.dateRange?.start?.toLocaleDateString()} to {dataSummary.dateRange?.end?.toLocaleDateString()}
        </p>
      </div>

      {/* Control Panel */}
      <div className="control-panel">
        <div className="control-grid">
          <div className="control-item">
            <label className="control-label">Plant</label>
            <select 
              value={filters.plant} 
              onChange={(e) => updateFilter('plant', e.target.value)}
              className="control-select"
            >
              <option value="all">All Plants</option>
              {dataSummary.plants?.map(plant => (
                <option key={plant.value} value={plant.value}>
                  {plant.label}
                </option>
              ))}
            </select>
          </div>
          <div className="control-item">
            <label className="control-label">Panel Size</label>
            <select 
              value={filters.panel_size} 
              onChange={(e) => updateFilter('panel_size', e.target.value)}
              className="control-select"
            >
              <option value="all">All Sizes</option>
              {dataSummary.panelSizes?.map(size => (
                <option key={size.value} value={size.value}>
                  {size.label}
                </option>
              ))}
            </select>
          </div>
          <div className="control-item">
            <label className="control-label">Market Segment</label>
            <select 
              value={filters.market_segment} 
              onChange={(e) => updateFilter('market_segment', e.target.value)}
              className="control-select"
            >
              <option value="all">All Segments</option>
              {dataSummary.marketSegments?.map(segment => (
                <option key={segment.value} value={segment.value}>
                  {segment.label}
                </option>
              ))}
            </select>
          </div>
          <div className="control-item">
            <label className="control-label">Time Range</label>
            <select 
              value={filters.date_range} 
              onChange={(e) => updateFilter('date_range', e.target.value)}
              className="control-select"
            >
              <option value="all">All Data</option>
              <option value="1month">Last Month</option>
              <option value="3months">Last 3 Months</option>
              <option value="6months">Last 6 Months</option>
              <option value="1year">Last Year</option>
            </select>
          </div>
        </div>
        <div className="control-actions">
          <button onClick={resetFilters} className="control-button">
            Reset Filters
          </button>
          <button onClick={refreshData} className="control-button control-button-primary">
            <RefreshCw className="button-icon" />
            Refresh Data
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <MetricCard
          title="Total Revenue"
          value={`$${(kpis.totalRevenue / 1000000).toFixed(1)}M`}
          icon={DollarSign}
          color="bg-green"
        />
        <MetricCard
          title="Average Yield"
          value={`${(kpis.avgYield * 100).toFixed(1)}%`}
          icon={TrendingUp}
          color="bg-blue"
        />
        <MetricCard
          title="Capacity Utilization"
          value={`${(kpis.avgCapacity * 100).toFixed(1)}%`}
          icon={Package}
          color="bg-purple"
        />
        <MetricCard
          title="Supply Disruptions"
          value={kpis.disruptions.toString()}
          icon={AlertTriangle}
          color="bg-red"
        />
      </div>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <TabButton id="overview" label="Overview" icon={BarChart3} />
        <TabButton id="production" label="Production" icon={Package} />
        <TabButton id="quality" label="Quality" icon={Activity} />
        <TabButton id="financial" label="Financial" icon={DollarSign} />
        <TabButton id="supply" label="Supply Chain" icon={AlertTriangle} />
        <TabButton id="trends" label="Trends" icon={TrendingUp} />
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <>
            <div className="chart-grid">
              <div className="chart-card">
                <h3 className="chart-title">Weekly Revenue Trends</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={weeklyTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="week" 
                      tick={{ fontSize: 12 }}
                      interval="preserveStartEnd"
                    />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']}
                      labelFormatter={(label) => `Week: ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="revenue" 
                      stroke="#8884d8" 
                      strokeWidth={2}
                      dot={{ fill: '#8884d8', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3 className="chart-title">Plant Performance Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={plantPerformance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="plant" />
                    <YAxis />
                    <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Yield']} />
                    <Bar dataKey="yield" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-grid">
              <div className="chart-card">
                <h3 className="chart-title">Panel Size Revenue Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={sizeAnalysis}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({size, percent}) => `${size} (${(percent * 100).toFixed(0)}%)`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="revenue"
                    >
                      {sizeAnalysis.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3 className="chart-title">Production vs Demand</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={weeklyTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" tick={{ fontSize: 12 }} />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="demand" 
                      stackId="1" 
                      stroke="#ffc658" 
                      fill="#ffc658" 
                      fillOpacity={0.6}
                      name="Demand"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="production" 
                      stackId="2" 
                      stroke="#8884d8" 
                      fill="#8884d8" 
                      fillOpacity={0.6}
                      name="Production"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}

        {activeTab === 'production' && (
          <div className="chart-grid">
            <div className="chart-card">
              <h3 className="chart-title">Capacity Utilization by Plant</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={plantPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="plant" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Capacity']} />
                  <Bar dataKey="capacity" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3 className="chart-title">Production Volume by Size</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={sizeAnalysis}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="size" />
                  <YAxis />
                  <Tooltip formatter={(value) => [value.toLocaleString(), 'Units']} />
                  <Bar dataKey="volume" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'quality' && (
          <div className="chart-grid">
            <div className="chart-card">
              <h3 className="chart-title">Quality Metrics Radar</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={plantPerformance}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="plant" />
                  <PolarRadiusAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                  <Radar 
                    name="Yield" 
                    dataKey="yield" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.3} 
                  />
                  <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`]} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3 className="chart-title">Defect Rate by Plant</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={plantPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="plant" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Defect Rate']} />
                  <Bar dataKey="defects" fill="#ff7300" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'financial' && (
          <div className="chart-grid">
            <div className="chart-card">
              <h3 className="chart-title">Revenue by Plant</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={plantPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="plant" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']} />
                  <Bar dataKey="revenue" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3 className="chart-title">Average Price by Panel Size</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={sizeAnalysis}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="size" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Avg Price']} />
                  <Bar dataKey="avg_price" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'supply' && (
          <div className="chart-grid">
            <div className="chart-card">
              <h3 className="chart-title">Supply Disruption Rate by Plant</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={supplyChainData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="plant" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Disruption Rate']} />
                  <Bar dataKey="disruption_rate" fill="#ff7300" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3 className="chart-title">On-Time Delivery Performance</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={supplyChainData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="plant" />
                  <YAxis domain={[80, 100]} />
                  <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'On-Time Delivery']} />
                  <Bar dataKey="avg_delivery" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'trends' && (
          <div className="full-width-chart">
            <div className="chart-card">
              <h3 className="chart-title">Multi-Metric Trends Over Time</h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={weeklyTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line 
                    yAxisId="left" 
                    type="monotone" 
                    dataKey="revenue" 
                    stroke="#8884d8" 
                    name="Revenue ($)" 
                    strokeWidth={2}
                  />
                  <Line 
                    yAxisId="right" 
                    type="monotone" 
                    dataKey="avg_yield" 
                    stroke="#82ca9d" 
                    name="Avg Yield" 
                    strokeWidth={2}
                  />
                  <Line 
                    yAxisId="right" 
                    type="monotone" 
                    dataKey="avg_capacity" 
                    stroke="#ffc658" 
                    name="Avg Capacity" 
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Data Summary Panel */}
      <div className="info-panel">
        <h4 className="info-title">Dataset Information</h4>
        <div className="data-summary-grid">
          <div className="data-summary-item">
            <span className="data-summary-label">Records:</span>
            <span className="data-summary-value">{dataSummary.mainRecords?.toLocaleString()}</span>
          </div>
          <div className="data-summary-item">
            <span className="data-summary-label">Date Range:</span>
            <span className="data-summary-value">
              {dataSummary.dateRange?.totalWeeks} weeks
            </span>
          </div>
          <div className="data-summary-item">
            <span className="data-summary-label">Panel Sizes:</span>
            <span className="data-summary-value">{dataSummary.panelSizes?.length} types</span>
          </div>
          <div className="data-summary-item">
            <span className="data-summary-label">Plants:</span>
            <span className="data-summary-value">{dataSummary.plants?.length} locations</span>
          </div>
        </div>
        <ul className="info-list">
          <li>• Real-time data loading from CSV files in /data folder</li>
          <li>• Interactive filtering and aggregation capabilities</li>
          <li>• Automatic data refresh and caching</li>
          <li>• Export-ready visualizations for research presentations</li>
        </ul>
      </div>
    </div>
  );
};

export default TFTLCDDashboard;