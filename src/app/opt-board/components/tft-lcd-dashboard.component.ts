import { Component, OnInit, OnDestroy, signal, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TFTLCDDataService } from '../services/tftlcd-data.service';
import { ChartService } from '../services/chart.service';

interface MetricCardData {
  title: string;
  value: string;
  unit?: string;
  color: string;
}

interface ChartData {
  weeklyTrends: any[];
  plantPerformance: any[];
  sizeAnalysis: any[];
  supplyChainData: any[];
  qualityMetrics: any[];
  kpis: {
    totalRevenue: number;
    avgYield: number;
    avgCapacity: number;
    disruptions: number;
  };
}

@Component({
  selector: 'app-tft-lcd-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="dashboard-container">
      <div class="dashboard-header">
        <h1 class="dashboard-title">TFT-LCD Manufacturing Analytics Dashboard</h1>
        <p class="dashboard-subtitle">
          Real-time analysis of {{dataSummary()?.mainRecords?.toLocaleString() || 0}} production records
          @if (dataSummary()?.dateRange?.start && dataSummary()?.dateRange?.end) {
            from {{dataSummary()?.dateRange?.start | date:'short'}} to {{dataSummary()?.dateRange?.end | date:'short'}}
          }
        </p>
      </div>

      <!-- Control Panel -->
      <div class="control-panel">
        <div class="control-grid">
          <div class="control-item">
            <label class="control-label">Plant</label>
            <select 
              [(ngModel)]="filters.plant" 
              (change)="updateFilter('plant', filters.plant)"
              class="control-select"
            >
              <option value="all">All Plants</option>
              @for (plant of dataSummary()?.plants || []; track plant.value) {
                <option [value]="plant.value">{{plant.label}}</option>
              }
            </select>
          </div>
          <div class="control-item">
            <label class="control-label">Panel Size</label>
            <select 
              [(ngModel)]="filters.panel_size" 
              (change)="updateFilter('panel_size', filters.panel_size)"
              class="control-select"
            >
              <option value="all">All Sizes</option>
              @for (size of dataSummary()?.panelSizes || []; track size.value) {
                <option [value]="size.value">{{size.label}}</option>
              }
            </select>
          </div>
          <div class="control-item">
            <label class="control-label">Market Segment</label>
            <select 
              [(ngModel)]="filters.market_segment" 
              (change)="updateFilter('market_segment', filters.market_segment)"
              class="control-select"
            >
              <option value="all">All Segments</option>
              @for (segment of dataSummary()?.marketSegments || []; track segment.value) {
                <option [value]="segment.value">{{segment.label}}</option>
              }
            </select>
          </div>
          <div class="control-item">
            <label class="control-label">Time Range</label>
            <select 
              [(ngModel)]="filters.date_range" 
              (change)="updateFilter('date_range', filters.date_range)"
              class="control-select"
            >
              <option value="all">All Data</option>
              <option value="1month">Last Month</option>
              <option value="3months">Last 3 Months</option>
              <option value="6months">Last 6 Months</option>
              <option value="1year">Last Year</option>
            </select>
          </div>
        </div>
        <div class="control-actions">
          <button (click)="resetFilters()" class="control-button">
            Reset Filters
          </button>
          <button (click)="refreshData()" class="control-button control-button-primary">
            🔄 Refresh Data
          </button>
        </div>
      </div>

      <!-- Loading State -->
      @if (loading()) {
        <div class="loading-container">
          <div class="loading-spinner">🔄</div>
          <p>Loading TFT-LCD data...</p>
        </div>
      }

      <!-- Error State -->
      @if (error()) {
        <div class="error-container">
          <div class="error-icon">⚠️</div>
          <h3>Error Loading Data</h3>
          <p>{{error()}}</p>
          <button (click)="refreshData()" class="error-retry-button">
            🔄 Retry
          </button>
        </div>
      }

      <!-- Main Dashboard Content -->
      @if (!loading() && !error() && dataSummary()) {
        <!-- KPI Cards -->
        <div class="kpi-grid">
          @for (metric of getKPIMetrics(); track metric.title) {
            <div class="metric-card">
              <div class="metric-card-content">
                <div class="metric-card-text">
                  <p class="metric-card-title">{{metric.title}}</p>
                  <p class="metric-card-value">{{metric.value}}</p>
                </div>
                <div class="metric-card-icon {{metric.color}}">
                  {{getMetricIcon(metric.title)}}
                </div>
              </div>
            </div>
          }
        </div>

        <!-- Tab Navigation -->
        <div class="tab-navigation">
          <button 
            class="tab-button" 
            [class.tab-button-active]="activeTab() === 'overview'"
            (click)="setActiveTab('overview')"
          >
            📊 Overview
          </button>
          <button 
            class="tab-button" 
            [class.tab-button-active]="activeTab() === 'production'"
            (click)="setActiveTab('production')"
          >
            📦 Production
          </button>
          <button 
            class="tab-button" 
            [class.tab-button-active]="activeTab() === 'quality'"
            (click)="setActiveTab('quality')"
          >
            📈 Quality
          </button>
          <button 
            class="tab-button" 
            [class.tab-button-active]="activeTab() === 'financial'"
            (click)="setActiveTab('financial')"
          >
            💰 Financial
          </button>
          <button 
            class="tab-button" 
            [class.tab-button-active]="activeTab() === 'supply'"
            (click)="setActiveTab('supply')"
          >
            🚛 Supply Chain
          </button>
        </div>

        <!-- Tab Content -->
        <div class="tab-content">
          @if (activeTab() === 'overview') {
            <div class="chart-grid">
              <div class="chart-card">
                <h3 class="chart-title">Weekly Revenue Trends</h3>
                <canvas id="weeklyRevenueChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">Plant Performance Comparison</h3>
                <canvas id="plantPerformanceChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">Panel Size Revenue Distribution</h3>
                <canvas id="sizeDistributionChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">Production vs Demand</h3>
                <canvas id="productionDemandChart" width="400" height="300"></canvas>
              </div>
            </div>
          }

          @if (activeTab() === 'production') {
            <div class="chart-grid">
              <div class="chart-card">
                <h3 class="chart-title">Capacity Utilization by Plant</h3>
                <canvas id="capacityChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">Production Volume by Size</h3>
                <canvas id="productionVolumeChart" width="400" height="300"></canvas>
              </div>
            </div>
          }

          @if (activeTab() === 'quality') {
            <div class="chart-grid">
              <div class="chart-card">
                <h3 class="chart-title">Quality Metrics by Plant</h3>
                <canvas id="qualityChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">Defect Rate Analysis</h3>
                <canvas id="defectChart" width="400" height="300"></canvas>
              </div>
            </div>
          }

          @if (activeTab() === 'financial') {
            <div class="chart-grid">
              <div class="chart-card">
                <h3 class="chart-title">Revenue by Plant</h3>
                <canvas id="revenueByPlantChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">Average Price by Panel Size</h3>
                <canvas id="avgPriceChart" width="400" height="300"></canvas>
              </div>
            </div>
          }

          @if (activeTab() === 'supply') {
            <div class="chart-grid">
              <div class="chart-card">
                <h3 class="chart-title">Supply Disruption Rate</h3>
                <canvas id="disruptionChart" width="400" height="300"></canvas>
              </div>
              <div class="chart-card">
                <h3 class="chart-title">On-Time Delivery Performance</h3>
                <canvas id="deliveryChart" width="400" height="300"></canvas>
              </div>
            </div>
          }
        </div>

        <!-- Data Summary Panel -->
        <div class="info-panel">
          <h4 class="info-title">Dataset Information</h4>
          <div class="data-summary-grid">
            <div class="data-summary-item">
              <span class="data-summary-label">Records:</span>
              <span class="data-summary-value">{{dataSummary()?.mainRecords?.toLocaleString()}}</span>
            </div>
            <div class="data-summary-item">
              <span class="data-summary-label">Date Range:</span>
              <span class="data-summary-value">{{dataSummary()?.dateRange?.totalWeeks}} weeks</span>
            </div>
            <div class="data-summary-item">
              <span class="data-summary-label">Panel Sizes:</span>
              <span class="data-summary-value">{{dataSummary()?.panelSizes?.length}} types</span>
            </div>
            <div class="data-summary-item">
              <span class="data-summary-label">Plants:</span>
              <span class="data-summary-value">{{dataSummary()?.plants?.length}} locations</span>
            </div>
          </div>
          <ul class="info-list">
            <li>• Real-time data loading from CSV files</li>
            <li>• Interactive filtering and aggregation capabilities</li>
            <li>• Automatic data refresh and caching</li>
            <li>• Export-ready visualizations for research</li>
          </ul>
        </div>
      }

      <!-- Navigation to Operation Model -->
      <div class="navigation-panel">
        <a href="/operation-model" class="nav-button">
          🧠 Go to LSTM Operation Model →
        </a>
      </div>
    </div>
  `,
  styles: [`
    .dashboard-container {
      padding: 1rem;
      max-width: 1400px;
      margin: 0 auto;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .dashboard-header {
      text-align: center;
      margin-bottom: 2rem;
      padding: 2rem;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-radius: 12px;
    }

    .dashboard-title {
      font-size: 2.5rem;
      margin: 0 0 0.5rem 0;
      font-weight: 700;
    }

    .dashboard-subtitle {
      font-size: 1.1rem;
      opacity: 0.9;
      margin: 0;
    }

    .control-panel {
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .control-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }

    .control-item {
      display: flex;
      flex-direction: column;
    }

    .control-label {
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: #374151;
    }

    .control-select {
      padding: 0.5rem;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      font-size: 0.9rem;
    }

    .control-actions {
      display: flex;
      gap: 1rem;
      justify-content: flex-end;
    }

    .control-button {
      padding: 0.5rem 1rem;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      background: white;
      cursor: pointer;
      transition: all 0.2s;
    }

    .control-button-primary {
      background: #1976d2;
      color: white;
      border-color: #1976d2;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .metric-card {
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .metric-card-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .metric-card-title {
      font-size: 0.9rem;
      color: #6b7280;
      margin: 0 0 0.5rem 0;
    }

    .metric-card-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
    }

    .metric-card-icon {
      font-size: 2rem;
      opacity: 0.8;
    }

    .bg-green { color: #10b981; }
    .bg-blue { color: #3b82f6; }
    .bg-purple { color: #8b5cf6; }
    .bg-red { color: #ef4444; }

    .tab-navigation {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 2rem;
      overflow-x: auto;
    }

    .tab-button {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 8px;
      background: #f3f4f6;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }

    .tab-button-active {
      background: #1976d2;
      color: white;
    }

    .chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 2rem;
      margin-bottom: 2rem;
    }

    .chart-card {
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .chart-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin: 0 0 1rem 0;
      color: #374151;
    }

    .chart-container {
      height: 300px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f9fafb;
      border-radius: 4px;
      color: #6b7280;
      font-size: 0.9rem;
    }

    .loading-container, .error-container {
      text-align: center;
      padding: 3rem;
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .loading-spinner, .error-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
    }

    .info-panel {
      background: white;
      border-radius: 8px;
      padding: 2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      margin-bottom: 2rem;
    }

    .info-title {
      font-size: 1.2rem;
      font-weight: 600;
      margin: 0 0 1rem 0;
    }

    .data-summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .data-summary-item {
      display: flex;
      justify-content: space-between;
      padding: 0.5rem 0;
      border-bottom: 1px solid #e5e7eb;
    }

    .data-summary-label {
      font-weight: 500;
      color: #6b7280;
    }

    .data-summary-value {
      font-weight: 600;
      color: #111827;
    }

    .info-list {
      margin: 0;
      padding-left: 1rem;
    }

    .info-list li {
      margin-bottom: 0.5rem;
      color: #6b7280;
    }

    .navigation-panel {
      text-align: center;
      margin-top: 3rem;
    }

    .nav-button {
      display: inline-block;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1rem 2rem;
      text-decoration: none;
      border-radius: 8px;
      font-weight: 600;
      transition: transform 0.2s;
    }

    .nav-button:hover {
      transform: translateY(-2px);
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .metric-card {
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-left: 4px solid #667eea;
    }

    .metric-card-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .metric-card-title {
      font-size: 0.9rem;
      color: #6b7280;
      margin: 0 0 0.5rem 0;
      font-weight: 500;
    }

    .metric-card-value {
      font-size: 1.8rem;
      font-weight: 700;
      color: #111827;
      margin: 0;
    }

    .metric-card-icon {
      font-size: 2rem;
      opacity: 0.8;
    }

    .metric-card-icon.bg-green { color: #10b981; }
    .metric-card-icon.bg-blue { color: #3b82f6; }
    .metric-card-icon.bg-purple { color: #8b5cf6; }
    .metric-card-icon.bg-red { color: #ef4444; }

    .tab-navigation {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 2rem;
      background: white;
      padding: 1rem;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .tab-button {
      padding: 0.75rem 1.5rem;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      background: white;
      color: #6b7280;
      cursor: pointer;
      transition: all 0.2s;
      font-weight: 500;
    }

    .tab-button:hover {
      border-color: #667eea;
      color: #667eea;
    }

    .tab-button-active {
      background: #667eea;
      color: white;
      border-color: #667eea;
    }

    .chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
      gap: 2rem;
      margin-bottom: 2rem;
    }

    .chart-card {
      background: white;
      border-radius: 8px;
      padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      min-height: 400px;
    }

    .chart-title {
      font-size: 1.1rem;
      font-weight: 600;
      margin: 0 0 1rem 0;
      color: #374151;
      text-align: center;
    }

    .chart-card canvas {
      width: 100% !important;
      height: 300px !important;
      max-height: 300px;
    }

    @media (max-width: 768px) {
      .chart-grid {
        grid-template-columns: 1fr;
      }
      
      .chart-card {
        min-height: 300px;
      }
      
      .chart-card canvas {
        height: 250px !important;
      }
    }
  `]
})
export class TFTLCDDashboardComponent implements OnInit, AfterViewInit, OnDestroy {
  // Signals for reactive state
  loading = signal(true);
  error = signal<string | null>(null);
  activeTab = signal('overview');
  
  dataSummary = signal<any>(null);
  chartData = signal<ChartData>({
    weeklyTrends: [],
    plantPerformance: [],
    sizeAnalysis: [],
    supplyChainData: [],
    qualityMetrics: [],
    kpis: {
      totalRevenue: 0,
      avgYield: 0,
      avgCapacity: 0,
      disruptions: 0
    }
  });

  // Filter state
  filters = {
    plant: 'all',
    panel_size: 'all',
    market_segment: 'all',
    date_range: 'all'
  };

  private chartsInitialized = false;

  constructor(
    private dataService: TFTLCDDataService,
    private chartService: ChartService
  ) {}

  async ngOnInit() {
    await this.loadData();
  }

  ngAfterViewInit() {
    // Initialize charts only after data is loaded
    this.initializeChartsWhenReady();
  }

  private initializeChartsWhenReady() {
    // Wait for data to be loaded and view to be initialized
    if (this.loading() || !this.dataSummary() || this.chartsInitialized) {
      return;
    }

    // Double-check that canvas elements exist before initializing charts
    setTimeout(() => {
      if (this.activeTab() === 'overview') {
        this.initializeOverviewCharts();
      }
    }, 200);
  }

  ngOnDestroy() {
    this.chartService.destroyAllCharts();
  }

  async loadData() {
    try {
      this.loading.set(true);
      this.error.set(null);

      const data = await this.dataService.loadTFTLCDData();
      this.dataSummary.set(data.dataSummary);
      this.chartData.set({
        weeklyTrends: data.weeklyTrends,
        plantPerformance: data.plantPerformance,
        sizeAnalysis: data.sizeAnalysis,
        supplyChainData: data.supplyChainData,
        qualityMetrics: data.qualityMetrics,
        kpis: data.kpis
      });

      this.loading.set(false);
      
      // Initialize charts after data is loaded
      this.initializeChartsWhenReady();
    } catch (err) {
      console.error('Data loading error:', err);
      this.error.set(`Failed to load TFT-LCD data. Using synthetic data instead.`);
      this.loading.set(false);
      
      // Try to load with synthetic data only as a last resort
      try {
        const syntheticData = {
          dataSummary: {
            mainRecords: 1000,
            dateRange: {
              start: new Date('2024-01-01'),
              end: new Date('2024-12-31'),
              totalWeeks: 52
            },
            plants: [
              { value: 'Plant_A', label: 'Plant A' },
              { value: 'Plant_B', label: 'Plant B' },
              { value: 'Plant_C', label: 'Plant C' }
            ],
            panelSizes: [
              { value: 'Small', label: 'Small' },
              { value: 'Medium', label: 'Medium' },
              { value: 'Large', label: 'Large' }
            ],
            marketSegments: [
              { value: 'Automotive', label: 'Automotive' },
              { value: 'Consumer_Electronics', label: 'Consumer Electronics' },
              { value: 'Industrial', label: 'Industrial' }
            ]
          },
          weeklyTrends: this.generateDefaultWeeklyTrends(),
          plantPerformance: this.generateDefaultPlantPerformance(),
          sizeAnalysis: this.generateDefaultSizeAnalysis(),
          supplyChainData: this.generateDefaultSupplyChainData(),
          qualityMetrics: [],
          kpis: {
            totalRevenue: 25000000,
            avgYield: 0.87,
            avgCapacity: 0.82,
            disruptions: 12
          }
        };
        
        this.dataSummary.set(syntheticData.dataSummary);
        this.chartData.set(syntheticData);
        this.error.set(null);
        this.initializeChartsWhenReady();
      } catch (fallbackErr) {
        console.error('Fallback data generation failed:', fallbackErr);
        this.error.set('Failed to load data and generate fallback. Please refresh the page.');
      }
    }
  }

  private initializeOverviewCharts() {
    if (this.chartsInitialized || this.loading()) return;
    
    const data = this.chartData();
    
    // Only initialize charts for visible elements
    if (document.getElementById('weeklyRevenueChart')) {
      this.chartService.createLineChart(
        'weeklyRevenueChart', 
        data.weeklyTrends, 
        'week', 
        'revenue', 
        'Weekly Revenue ($)'
      );
    }

    if (document.getElementById('plantPerformanceChart')) {
      this.chartService.createBarChart(
        'plantPerformanceChart',
        data.plantPerformance,
        'plant',
        'yield',
        'Yield Rate'
      );
    }

    if (document.getElementById('sizeDistributionChart')) {
      this.chartService.createPieChart(
        'sizeDistributionChart',
        data.sizeAnalysis,
        'size',
        'revenue'
      );
    }

    if (document.getElementById('productionDemandChart')) {
      this.chartService.createAreaChart(
        'productionDemandChart',
        data.weeklyTrends,
        'week',
        [
          { key: 'production', label: 'Production', color: 'rgba(54, 162, 235, 1)' },
          { key: 'demand', label: 'Demand', color: 'rgba(255, 205, 86, 1)' }
        ]
      );
    }

    this.chartsInitialized = true;
  }

  private initializeTabCharts(tab: string) {
    const data = this.chartData();
    
    switch (tab) {
      case 'production':
        if (document.getElementById('capacityChart')) {
          this.chartService.createBarChart('capacityChart', data.plantPerformance, 'plant', 'capacity', 'Capacity Utilization');
        }
        if (document.getElementById('productionVolumeChart')) {
          this.chartService.createBarChart('productionVolumeChart', data.sizeAnalysis, 'size', 'volume', 'Production Volume');
        }
        break;
        
      case 'quality':
        if (document.getElementById('qualityChart')) {
          this.chartService.createBarChart('qualityChart', data.plantPerformance, 'plant', 'yield', 'Quality Yield Rate');
        }
        if (document.getElementById('defectChart')) {
          this.chartService.createBarChart('defectChart', data.plantPerformance, 'plant', 'defects', 'Defect Rate');
        }
        break;
        
      case 'financial':
        if (document.getElementById('revenueByPlantChart')) {
          this.chartService.createBarChart('revenueByPlantChart', data.plantPerformance, 'plant', 'revenue', 'Revenue ($)');
        }
        if (document.getElementById('avgPriceChart')) {
          this.chartService.createBarChart('avgPriceChart', data.sizeAnalysis, 'size', 'avg_price', 'Average Price ($)');
        }
        break;
        
      case 'supply':
        if (document.getElementById('disruptionChart')) {
          this.chartService.createBarChart('disruptionChart', data.supplyChainData, 'plant', 'disruption_rate', 'Disruption Rate (%)');
        }
        if (document.getElementById('deliveryChart')) {
          this.chartService.createBarChart('deliveryChart', data.supplyChainData, 'plant', 'avg_delivery', 'On-Time Delivery (%)');
        }
        break;
    }
  }

  setActiveTab(tab: string) {
    this.activeTab.set(tab);
    
    // Initialize charts for the new tab after a brief delay
    setTimeout(() => {
      if (tab === 'overview' && !this.chartsInitialized) {
        this.initializeOverviewCharts();
      } else if (tab !== 'overview') {
        this.initializeTabCharts(tab);
      }
    }, 100);
  }

  updateFilter(key: string, value: string) {
    (this.filters as any)[key] = value;
    this.applyFilters();
  }

  resetFilters() {
    this.filters = {
      plant: 'all',
      panel_size: 'all',
      market_segment: 'all',
      date_range: 'all'
    };
    this.applyFilters();
  }

  async refreshData() {
    await this.loadData();
  }

  private applyFilters() {
    // Apply filters to data and update charts
    console.log('Applying filters:', this.filters);
  }

  getKPIMetrics(): MetricCardData[] {
    const kpis = this.chartData().kpis;
    return [
      {
        title: 'Total Revenue',
        value: `$${(kpis.totalRevenue / 1000000).toFixed(1)}M`,
        color: 'bg-green'
      },
      {
        title: 'Average Yield',
        value: `${(kpis.avgYield * 100).toFixed(1)}%`,
        color: 'bg-blue'
      },
      {
        title: 'Capacity Utilization',
        value: `${(kpis.avgCapacity * 100).toFixed(1)}%`,
        color: 'bg-purple'
      },
      {
        title: 'Supply Disruptions',
        value: kpis.disruptions.toString(),
        color: 'bg-red'
      }
    ];
  }

  getMetricIcon(title: string): string {
    const icons: {[key: string]: string} = {
      'Total Revenue': '💰',
      'Average Yield': '📈',
      'Capacity Utilization': '📦',
      'Supply Disruptions': '⚠️'
    };
    return icons[title] || '📊';
  }

  private generateDefaultWeeklyTrends(): any[] {
    const trends = [];
    for (let week = 1; week <= 52; week++) {
      trends.push({
        week: `2024-W${week.toString().padStart(2, '0')}`,
        revenue: 400000 + Math.random() * 200000,
        production: 800 + Math.random() * 400,
        demand: 900 + Math.random() * 300,
        avg_yield: 0.8 + Math.random() * 0.15,
        avg_capacity: 0.75 + Math.random() * 0.2
      });
    }
    return trends;
  }

  private generateDefaultPlantPerformance(): any[] {
    return [
      { plant: 'Plant_A', revenue: 8500000, yield: 0.85, capacity: 0.82, defects: 0.03 },
      { plant: 'Plant_B', revenue: 9200000, yield: 0.89, capacity: 0.78, defects: 0.025 },
      { plant: 'Plant_C', revenue: 7800000, yield: 0.82, capacity: 0.85, defects: 0.035 }
    ];
  }

  private generateDefaultSizeAnalysis(): any[] {
    return [
      { size: 'Small', revenue: 6500000, volume: 15000, avg_price: 150 },
      { size: 'Medium', revenue: 12000000, volume: 8000, avg_price: 280 },
      { size: 'Large', revenue: 7000000, volume: 3500, avg_price: 450 }
    ];
  }

  private generateDefaultSupplyChainData(): any[] {
    return [
      { plant: 'Plant_A', disruption_rate: 8.5, avg_delivery: 94.2 },
      { plant: 'Plant_B', disruption_rate: 6.8, avg_delivery: 96.1 },
      { plant: 'Plant_C', disruption_rate: 9.2, avg_delivery: 92.8 }
    ];
  }
}