// src/components/OptimizationAnalysisTools.js
import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend } from 'recharts';
import { Target, TrendingUp, Zap, Shield, DollarSign, Package, AlertTriangle, BarChart3, RefreshCw } from 'lucide-react';
import useOptimizationData from '../hooks/useOptimizationData';
import './OptimizationAnalysisTools.css';

const OptimizationAnalysisTools = () => {
  const [activeAnalysis, setActiveAnalysis] = useState('pareto');
  const [selectedObjectives, setSelectedObjectives] = useState(['cost', 'quality']);

  // Use the real optimization data hook
  const {
    paretoFronts,
    convergenceData,
    robustnessData,
    performanceData,
    radarData,
    tftData,
    loading,
    error,
    optimizationMethod,
    updateOptimizationMethod,
    refreshData
  } = useOptimizationData();

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ffb347'];

  const AnalysisCard = ({ title, description, isActive, onClick, icon: Icon }) => (
    <div 
      className={`analysis-card ${isActive ? 'analysis-card-active' : ''}`}
      onClick={onClick}
    >
      <div className="analysis-card-header">
        <Icon className={`analysis-card-icon ${isActive ? 'analysis-card-icon-active' : ''}`} />
        <h3 className={`analysis-card-title ${isActive ? 'analysis-card-title-active' : ''}`}>{title}</h3>
      </div>
      <p className={`analysis-card-description ${isActive ? 'analysis-card-description-active' : ''}`}>{description}</p>
    </div>
  );

  const ObjectiveSelector = () => (
    <div className="objective-selector">
      <h3 className="objective-selector-title">Analysis Configuration</h3>
      <div className="objective-selector-grid">
        <div className="objective-selector-item">
          <label className="objective-selector-label">Optimization Method</label>
          <select 
            value={optimizationMethod} 
            onChange={(e) => updateOptimizationMethod(e.target.value)}
            className="objective-selector-select"
          >
            <option value="all">All Methods</option>
            <option value="NSGA-II">NSGA-II</option>
            <option value="NSGA-III">NSGA-III</option>
            <option value="MOEA/D">MOEA/D</option>
            <option value="SPEA2">SPEA2</option>
            <option value="BMOO">BMOO</option>
            <option value="DMOL">DMOL</option>
            <option value="RSP">RSP</option>
          </select>
        </div>
        <div className="objective-selector-item">
          <label className="objective-selector-label">Primary Objective</label>
          <select className="objective-selector-select">
            <option value="cost">Cost Minimization</option>
            <option value="quality">Quality Maximization</option>
            <option value="service">Service Level</option>
            <option value="resilience">Resilience</option>
          </select>
        </div>
        <div className="objective-selector-item">
          <label className="objective-selector-label">Secondary Objective</label>
          <select className="objective-selector-select">
            <option value="quality">Quality Maximization</option>
            <option value="service">Service Level</option>
            <option value="resilience">Resilience</option>
            <option value="cost">Cost Minimization</option>
          </select>
        </div>
        <div className="objective-selector-item">
          <label className="objective-selector-label">Actions</label>
          <button onClick={refreshData} className="objective-refresh-button">
            <RefreshCw className="button-icon" />
            Refresh Data
          </button>
        </div>
      </div>
      
      {tftData && (
        <div className="data-source-info">
          <p className="data-source-text">
            Optimization scenarios generated from {tftData.summary?.mainRecords?.toLocaleString()} TFT-LCD production records
          </p>
        </div>
      )}
    </div>
  );

  const LoadingSpinner = () => (
    <div className="loading-container">
      <RefreshCw className="loading-spinner" />
      <p>Loading optimization analysis data...</p>
    </div>
  );

  const ErrorMessage = ({ error, onRetry }) => (
    <div className="error-container">
      <AlertTriangle className="error-icon" />
      <h3>Error Loading Optimization Data</h3>
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

  return (
    <div className="optimization-container">
      <div className="optimization-header">
        <h1 className="optimization-title">Multi-Objective Optimization Analysis</h1>
        <p className="optimization-subtitle">
          Advanced visualization tools for benchmarking hybrid optimization approaches (EMOO, BMOO, DMOL, RSP)
          {tftData && ` - Based on real TFT-LCD manufacturing data`}
        </p>
      </div>

      <ObjectiveSelector />

      {/* Analysis Type Selection */}
      <div className="analysis-grid">
        <AnalysisCard
          title="Pareto Front Analysis"
          description="Visualize trade-offs between competing objectives"
          isActive={activeAnalysis === 'pareto'}
          onClick={() => setActiveAnalysis('pareto')}
          icon={Target}
        />
        <AnalysisCard
          title="Convergence Analysis"
          description="Track algorithm convergence over generations"
          isActive={activeAnalysis === 'convergence'}
          onClick={() => setActiveAnalysis('convergence')}
          icon={TrendingUp}
        />
        <AnalysisCard
          title="Performance Comparison"
          description="Compare methods across multiple metrics"
          isActive={activeAnalysis === 'performance'}
          onClick={() => setActiveAnalysis('performance')}
          icon={BarChart3}
        />
        <AnalysisCard
          title="Robustness Analysis"
          description="Evaluate performance under uncertainty"
          isActive={activeAnalysis === 'robustness'}
          onClick={() => setActiveAnalysis('robustness')}
          icon={Shield}
        />
      </div>

      {/* Analysis Content */}
      <div className="analysis-content">
        {activeAnalysis === 'pareto' && (
          <>
            <div className="chart-container-grid">
              <div className="chart-container">
                <h3 className="chart-container-title">Cost vs Quality Trade-off</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="cost" 
                      type="number" 
                      name="Cost" 
                      domain={['dataMin - 10', 'dataMax + 10']}
                      label={{ value: 'Cost ($)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      dataKey="quality" 
                      type="number" 
                      name="Quality" 
                      domain={['dataMin - 5', 'dataMax + 5']}
                      label={{ value: 'Quality Score', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      formatter={(value, name) => [value.toFixed(2), name]}
                      labelFormatter={() => ''}
                    />
                    <Legend />
                    {optimizationMethod === 'all' ? (
                      ['NSGA-II', 'BMOO', 'DMOL', 'RSP'].map((method, index) => (
                        <Scatter
                          key={method}
                          name={method}
                          data={paretoFronts.filter(d => d.method === method)}
                          fill={COLORS[index]}
                        />
                      ))
                    ) : (
                      <Scatter
                        name={optimizationMethod}
                        data={paretoFronts}
                        fill="#8884d8"
                      />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-container">
                <h3 className="chart-container-title">Service Level vs Resilience</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="serviceLevel" 
                      type="number" 
                      name="Service Level" 
                      domain={[80, 105]}
                      label={{ value: 'Service Level (%)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      dataKey="resilience" 
                      type="number" 
                      name="Resilience" 
                      domain={[70, 105]}
                      label={{ value: 'Resilience Score', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      formatter={(value, name) => [value.toFixed(2), name]}
                      labelFormatter={() => ''}
                    />
                    <Legend />
                    {optimizationMethod === 'all' ? (
                      ['NSGA-II', 'BMOO', 'DMOL', 'RSP'].map((method, index) => (
                        <Scatter
                          key={method}
                          name={method}
                          data={paretoFronts.filter(d => d.method === method)}
                          fill={COLORS[index]}
                        />
                      ))
                    ) : (
                      <Scatter
                        name={optimizationMethod}
                        data={paretoFronts}
                        fill="#82ca9d"
                      />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="chart-container">
              <h3 className="chart-container-title">Multi-Objective Solution Space</h3>
              <div className="solutions-grid">
                {paretoFronts.slice(0, 9).map((point, index) => (
                  <div key={index} className="solution-card">
                    <div className="solution-method">{point.method}</div>
                    <div className="solution-details">
                      Cost: ${point.cost.toFixed(0)} | Quality: {point.quality.toFixed(1)} | 
                      Service: {point.serviceLevel.toFixed(1)}% | Resilience: {point.resilience.toFixed(1)}
                    </div>
                    <div className="solution-hypervolume">
                      Hypervolume: {point.hypervolume.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {activeAnalysis === 'convergence' && (
          <div className="chart-container-grid">
            <div className="chart-container">
              <h3 className="chart-container-title">Hypervolume Convergence</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={convergenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="generation" label={{ value: 'Generation', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Hypervolume', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  {optimizationMethod === 'all' ? (
                    ['NSGA-II', 'BMOO', 'DMOL', 'RSP'].map((method, index) => (
                      <Line
                        key={method}
                        type="monotone"
                        dataKey="hypervolume"
                        data={convergenceData.filter(d => d.method === method)}
                        stroke={COLORS[index]}
                        name={method}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))
                  ) : (
                    <Line
                      type="monotone"
                      dataKey="hypervolume"
                      stroke="#8884d8"
                      strokeWidth={2}
                      name="Hypervolume"
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h3 className="chart-container-title">Inverted Generational Distance (IGD)</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={convergenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="generation" label={{ value: 'Generation', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'IGD', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  {optimizationMethod === 'all' ? (
                    ['NSGA-II', 'BMOO', 'DMOL', 'RSP'].map((method, index) => (
                      <Line
                        key={method}
                        type="monotone"
                        dataKey="igd"
                        data={convergenceData.filter(d => d.method === method)}
                        stroke={COLORS[index]}
                        name={method}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))
                  ) : (
                    <Line
                      type="monotone"
                      dataKey="igd"
                      stroke="#82ca9d"
                      strokeWidth={2}
                      name="IGD"
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeAnalysis === 'performance' && (
          <div className="chart-container-grid">
            <div className="chart-container">
              <h3 className="chart-container-title">Performance Radar Chart</h3>
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="method" />
                  <PolarRadiusAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                  <Radar
                    name="Hypervolume"
                    dataKey="Hypervolume"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.3}
                  />
                  <Radar
                    name="IGD"
                    dataKey="IGD"
                    stroke="#82ca9d"
                    fill="#82ca9d"
                    fillOpacity={0.3}
                  />
                  <Tooltip />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h3 className="chart-container-title">Method Performance Comparison</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={performanceData.filter(d => d.metric === 'Hypervolume')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="method" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Performance']} />
                  <Bar dataKey="normalized" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeAnalysis === 'robustness' && (
          <div className="chart-container-grid">
            <div className="chart-container">
              <h3 className="chart-container-title">Value-at-Risk Analysis</h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={robustnessData.filter(d => d.scenario === 'High_Uncertainty')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="method" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="valueAtRisk" fill="#8884d8" name="Value at Risk" />
                  <Bar dataKey="conditionalVaR" fill="#82ca9d" name="Conditional VaR" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h3 className="chart-container-title">Uncertainty Impact Analysis</h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={robustnessData.filter(d => d.method === 'NSGA-II')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="scenario" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="expectedValue" 
                    stackId="1" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    name="Expected Value"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="variance" 
                    stackId="1" 
                    stroke="#82ca9d" 
                    fill="#82ca9d" 
                    name="Variance"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Research Insights Panel */}
      <div className="insights-panel">
        <h3 className="insights-title">Research Insights & Recommendations</h3>
        <div className="insights-grid">
          <div className="insight-card insight-blue">
            <h4 className="insight-card-title">EMOO Performance</h4>
            <p className="insight-card-text">
              NSGA-II shows consistent convergence across all objectives with good diversity maintenance. 
              Based on real TFT-LCD data patterns.
            </p>
          </div>
          <div className="insight-card insight-green">
            <h4 className="insight-card-title">BMOO Advantages</h4>
            <p className="insight-card-text">
              Bayesian methods excel in expensive function evaluations with superior sample efficiency. 
              Ideal for TFT-LCD manufacturing optimization.
            </p>
          </div>
          <div className="insight-card insight-purple">
            <h4 className="insight-card-title">DMOL Scalability</h4>
            <p className="insight-card-text">
              Deep learning approaches show promise for high-dimensional objective spaces found in 
              complex manufacturing systems.
            </p>
          </div>
          <div className="insight-card insight-orange">
            <h4 className="insight-card-title">RSP Robustness</h4>
            <p className="insight-card-text">
              Robust stochastic programming provides best worst-case performance guarantees under 
              supply chain uncertainties.
            </p>
          </div>
        </div>
        
        {tftData && (
          <div className="data-integration-info">
            <h4 className="integration-title">Real Data Integration</h4>
            <div className="integration-stats">
              <div className="integration-stat">
                <span className="stat-label">Source Records:</span>
                <span className="stat-value">{tftData.summary?.mainRecords?.toLocaleString()}</span>
              </div>
              <div className="integration-stat">
                <span className="stat-label">Time Period:</span>
                <span className="stat-value">{tftData.summary?.dateRange?.totalWeeks} weeks</span>
              </div>
              <div className="integration-stat">
                <span className="stat-label">Manufacturing Plants:</span>
                <span className="stat-value">{tftData.summary?.plants?.length}</span>
              </div>
              <div className="integration-stat">
                <span className="stat-label">Panel Sizes:</span>
                <span className="stat-value">{tftData.summary?.panelSizes?.length}</span>
              </div>
            </div>
            <p className="integration-description">
              Optimization scenarios are generated using realistic parameter ranges extracted from actual 
              TFT-LCD manufacturing performance data, ensuring practical relevance and validity.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default OptimizationAnalysisTools;