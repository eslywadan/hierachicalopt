// src/hooks/useTFTLCDData.js
import { useState, useEffect, useMemo } from 'react';

const useTFTLCDData = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    plant: 'all',
    panel_size: 'all',
    market_segment: 'all',
    date_range: 'all'
  });

  // Generate sample data for demonstration
  const generateSampleData = () => {
    const weeks = ['2023-W01', '2023-W02', '2023-W03', '2023-W04', '2023-W05', '2023-W06'];
    const plants = ['Taichung', 'Paju', 'Hefei', 'Guangzhou'];
    const sizes = ['32"', '43"', '55"', '65"', '75"'];
    
    return {
      weeklyTrends: weeks.map(week => ({
        week,
        revenue: Math.round(Math.random() * 500000 + 200000),
        production: Math.round(Math.random() * 2000 + 1000),
        demand: Math.round(Math.random() * 2200 + 1100),
        avg_yield: Math.random() * 0.1 + 0.85,
        avg_capacity: Math.random() * 0.2 + 0.7
      })),
      
      plantPerformance: plants.map(plant => ({
        plant,
        revenue: Math.round(Math.random() * 1000000 + 500000),
        production: Math.round(Math.random() * 5000 + 2000),
        yield: Math.random() * 0.1 + 0.85,
        capacity: Math.random() * 0.2 + 0.7,
        defects: Math.random() * 0.05 + 0.02
      })),
      
      sizeAnalysis: sizes.map(size => ({
        size,
        revenue: Math.round(Math.random() * 800000 + 200000),
        volume: Math.round(Math.random() * 3000 + 1000),
        avg_price: Math.round(Math.random() * 200 + 100),
        margin: Math.round(Math.random() * 100000 + 50000)
      })),
      
      supplyChainData: plants.map(plant => ({
        plant,
        disruption_rate: Math.random() * 5 + 2,
        avg_delivery: Math.random() * 10 + 85,
        disruptions: Math.round(Math.random() * 3),
        total: 50
      })),
      
      qualityMetrics: Array.from({ length: 20 }, (_, i) => ({
        id: i,
        yield: Math.random() * 10 + 85,
        defects: Math.random() * 3 + 1,
        delivery: Math.random() * 10 + 85,
        plant: plants[i % plants.length],
        size: [32, 43, 55, 65, 75][i % 5]
      }))
    };
  };

  const sampleData = useMemo(() => generateSampleData(), []);

  const kpis = useMemo(() => ({
    totalRevenue: sampleData.plantPerformance.reduce((sum, p) => sum + p.revenue, 0),
    avgYield: sampleData.plantPerformance.reduce((sum, p) => sum + p.yield, 0) / sampleData.plantPerformance.length,
    avgCapacity: sampleData.plantPerformance.reduce((sum, p) => sum + p.capacity, 0) / sampleData.plantPerformance.length,
    disruptions: sampleData.supplyChainData.reduce((sum, p) => sum + p.disruptions, 0)
  }), [sampleData]);

  const dataSummary = useMemo(() => ({
    mainRecords: 15840,
    dateRange: {
      start: new Date('2023-01-01'),
      end: new Date('2023-12-31'),
      totalWeeks: 52
    },
    plants: [
      { value: 'Plant_TW_Taichung', label: 'Taiwan - Taichung' },
      { value: 'Plant_KR_Paju', label: 'Korea - Paju' },
      { value: 'Plant_CN_Hefei', label: 'China - Hefei' },
      { value: 'Plant_CN_Guangzhou', label: 'China - Guangzhou' }
    ],
    panelSizes: [
      { value: '32', label: '32 inch' },
      { value: '43', label: '43 inch' },
      { value: '55', label: '55 inch' },
      { value: '65', label: '65 inch' },
      { value: '75', label: '75 inch' }
    ],
    marketSegments: [
      { value: 'TV', label: 'TV' },
      { value: 'Monitor', label: 'Monitor' },
      { value: 'Laptop', label: 'Laptop' },
      { value: 'Tablet', label: 'Tablet' }
    ]
  }), []);

  const updateFilter = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setFilters({
      plant: 'all',
      panel_size: 'all',
      market_segment: 'all',
      date_range: 'all'
    });
  };

  const refreshData = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      // In real implementation, this would reload data
    }, 1000);
  };

  return {
    // Data
    weeklyTrends: sampleData.weeklyTrends,
    plantPerformance: sampleData.plantPerformance,
    sizeAnalysis: sampleData.sizeAnalysis,
    supplyChainData: sampleData.supplyChainData,
    qualityMetrics: sampleData.qualityMetrics,
    kpis,
    dataSummary,
    filters,
    
    // State
    loading,
    error,
    
    // Actions
    updateFilter,
    resetFilters,
    refreshData
  };
};

export default useTFTLCDData;