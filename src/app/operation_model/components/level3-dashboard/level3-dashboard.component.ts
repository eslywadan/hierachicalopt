// ===FILE: src/app/operation_model/components/level3-dashboard/level3-dashboard.component.ts===

import { Component, OnInit, OnDestroy, AfterViewInit, signal, computed } from '@angular/core';
import { CommonModule, UpperCasePipe, DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
// Removed Material imports - using basic HTML for now
import { Subject, takeUntil } from 'rxjs';
import { Level3OperationService } from '../../services/level3-operation.service';
import { DataPoint, PredictionRequest, ValidationResult } from '../../models/level3-lstm-model';
import { Level3ChartComponent } from '../level3-chart/level3-chart.component';

@Component({
  selector: 'app-level3-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    Level3ChartComponent,
    UpperCasePipe,
    DatePipe,
    DecimalPipe
  ],
  templateUrl: './level3-dashboard.component.html',
  styleUrls: ['./level3-dashboard.component.scss']
})
export class Level3DashboardComponent implements OnInit, OnDestroy, AfterViewInit {
  private destroy$ = new Subject<void>();
  
  // Signals for state management
  trainingStatus = signal({ status: 'idle', progress: 0, message: '' });
  predictions = signal<DataPoint[]>([]);
  validationResult = signal<ValidationResult | null>(null);
  trainingData = signal<DataPoint[]>([]);
  
  // Computed signals
  isModelReady = computed(() => 
    this.trainingStatus().status === 'trained'
  );
  
  isTraining = computed(() => 
    this.trainingStatus().status === 'training'
  );
  
  // Form for prediction request
  predictionForm: FormGroup;
  
  // Configuration options (must match the model configuration)
  plants = ['Plant_1', 'Plant_2', 'Plant_3'];
  applications = ['Automotive', 'Consumer_Electronics', 'Industrial'];
  panelSizes = ['Small', 'Medium', 'Large'];
  
  // Table columns
  displayedColumns = ['date', 'wip', 'throughput', 'cycleTime', 'compliance'];
  
  // Chart type
  chartType: 'wip' | 'throughput' | 'cycleTime' | 'compliance' = 'wip';
  
  // Active tab
  activeTab: 'predictions' | 'visualizations' | 'data' | 'training' | 'littles-law' = 'predictions';

  // Training data visualization
  selectedPlant = 'all';
  selectedApplication = 'all';
  trainingChartType: 'wip' | 'throughput' | 'cycleTime' | 'compliance' = 'wip';
  filteredTrainingData = signal<DataPoint[]>([]);

  // Little's Law Analysis Charts
  private littlesLawChart: any = null;
  private complianceChart: any = null;

  constructor(
    private fb: FormBuilder,
    private level3Service: Level3OperationService
  ) {
    this.predictionForm = this.fb.group({
      plant: ['Plant_1', Validators.required],
      application: ['Automotive', Validators.required],
      panelSize: ['Medium', Validators.required],
      predictionDays: [7, [Validators.required, Validators.min(1), Validators.max(30)]],
      currentWIP: [1000, [Validators.required, Validators.min(0)]],
      plannedThroughput: [150, [Validators.required, Validators.min(0)]],
      targetProduction: [1000, Validators.min(0)]
    });
  }

  ngOnInit(): void {
    this.subscribeToServices();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    
    // Clean up charts
    if (this.littlesLawChart) {
      this.littlesLawChart.destroy();
    }
    if (this.complianceChart) {
      this.complianceChart.destroy();
    }
  }

  ngAfterViewInit(): void {
    // Charts will be initialized when Little's Law tab is first accessed
  }

  private subscribeToServices(): void {
    this.level3Service.getTrainingStatus()
      .pipe(takeUntil(this.destroy$))
      .subscribe(status => this.trainingStatus.set(status));

    this.level3Service.getPredictions()
      .pipe(takeUntil(this.destroy$))
      .subscribe(predictions => this.predictions.set(predictions));

    this.level3Service.getValidationResults()
      .pipe(takeUntil(this.destroy$))
      .subscribe(result => this.validationResult.set(result));
  }

  initializeModel(): void {
    this.level3Service.initializeModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          console.log('Model initialized successfully');
          this.loadTrainingData();
        },
        error: (error) => {
          console.error('Failed to initialize model');
          console.error(error);
        }
      });
  }

  trainModel(): void {
    this.level3Service.trainModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          console.log('Model trained successfully');
          console.log('Training metrics:', result.metrics);
          
          // Display evaluation results
          if (result.evaluation) {
            const evaluation = result.evaluation;
            console.log(`\n🎯 MODEL EVALUATION REPORT:`);
            console.log(`📊 Overall Feasibility: ${evaluation.feasibility}`);
            console.log(`🎯 R² Score: ${result.metrics.r2Score} (${evaluation.details.r2Assessment})`);
            console.log(`📈 MAPE: ${result.metrics.mape}% (${evaluation.details.mapeAssessment})`);
            console.log(`🔍 Confidence: ${evaluation.details.confidenceLevel}`);
            console.log(`💼 Recommended Use: ${evaluation.details.recommendedUse}`);
            console.log(`✅ Production Ready: ${evaluation.details.isUsableForProduction}`);
            
            // Show alert for poor performance
            if (evaluation.feasibility === 'POOR' || evaluation.feasibility === 'UNUSABLE') {
              console.warn('⚠️ MODEL PERFORMANCE WARNING: This model may not be suitable for production use.');
              console.warn('Consider retraining with improved parameters or more data.');
            }
          }
        },
        error: (error) => {
          console.error('Training failed');
          console.error(error);
        }
      });
  }

  makePrediction(): void {
    if (this.predictionForm.invalid) {
      console.warn('Please fill in all required fields');
      return;
    }

    const request: PredictionRequest = this.predictionForm.value;
    
    this.level3Service.makePrediction(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          console.log('Prediction completed');
          console.log('Analysis:', result.analysis);
        },
        error: (error) => {
          console.error('Prediction failed');
          console.error(error);
        }
      });
  }

  loadTrainingData(): void {
    try {
      const data = this.level3Service.exportTrainingData();
      console.log('📊 Total training data available:', data.length);
      
      // Load all data to ensure all plants are represented
      this.trainingData.set(data); // Load ALL data, not just first 1000
      this.filteredTrainingData.set(data); // Initialize with all data
      
      // Debug: Count per plant and get unique plant names
      const plantCounts: { [key: string]: number } = {};
      const uniquePlants = new Set<string>();
      
      data.forEach(d => {
        plantCounts[d.plant] = (plantCounts[d.plant] || 0) + 1;
        uniquePlants.add(d.plant);
      });
      
      console.log('📈 Training data loaded per plant:', plantCounts);
      console.log('🏭 Unique plant names found:', Array.from(uniquePlants));
      console.log('🔧 Configured plants:', this.plants);
      
      // Check if there's a mismatch
      const missingPlants = this.plants.filter(p => !uniquePlants.has(p));
      if (missingPlants.length > 0) {
        console.error('⚠️ These plants have no data:', missingPlants);
      }
    } catch (error) {
      console.error('Failed to load training data:', error);
    }
  }

  saveModel(): void {
    this.level3Service.saveModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => console.log('Model saved successfully'),
        error: () => console.error('Failed to save model')
      });
  }

  loadModel(): void {
    this.level3Service.loadModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          console.log('Model loaded successfully');
          this.loadTrainingData();
        },
        error: () => console.error('Failed to load model')
      });
  }

  exportData(): void {
    const data = this.predictions();
    if (data.length === 0) {
      console.warn('No data to export');
      return;
    }

    const csv = this.convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'level3_predictions.csv';
    link.click();
    window.URL.revokeObjectURL(url);
  }

  private convertToCSV(data: DataPoint[]): string {
    const headers = ['Date', 'Plant', 'Application', 'Panel Size', 'WIP', 'Throughput', 'Cycle Time', 'Finished Goods', 'Semi-Finished Goods'];
    const rows = data.map(d => [
      d.date.toISOString(),
      d.plant,
      d.application,
      d.panelSize,
      d.wip,
      d.throughput,
      d.cycleTime,
      d.finishedGoods,
      d.semiFinishedGoods
    ]);

    return [headers, ...rows].map(row => row.join(',')).join('\n');
  }

  // Training data visualization methods
  filterTrainingData(): void {
    const allTrainingData = this.trainingData();
    let filtered = allTrainingData;

    if (this.selectedPlant !== 'all') {
      filtered = filtered.filter(d => d.plant === this.selectedPlant);
    }

    if (this.selectedApplication !== 'all') {
      filtered = filtered.filter(d => d.application === this.selectedApplication);
    }

    this.filteredTrainingData.set(filtered);
  }

  updateTrainingChart(): void {
    // Trigger chart update by filtering data again
    this.filterTrainingData();
  }

  exportTrainingData(): void {
    const data = this.filteredTrainingData();
    if (data.length === 0) {
      console.warn('No training data to export');
      return;
    }

    const csv = this.convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `training_data_${this.selectedPlant}_${this.selectedApplication}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }

  getDateRange(): string {
    const data = this.filteredTrainingData();
    if (data.length === 0) return 'No data';
    
    const dates = data.map(d => d.date).sort();
    const start = dates[0].toLocaleDateString();
    const end = dates[dates.length - 1].toLocaleDateString();
    return `${start} - ${end}`;
  }

  getUniquePlants(): string[] {
    return Array.from(new Set(this.filteredTrainingData().map(d => d.plant)));
  }

  getUniqueApplications(): string[] {
    return Array.from(new Set(this.filteredTrainingData().map(d => d.application)));
  }

  getDayOfWeekName(date: Date): string {
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    return days[date.getDay()];
  }

  isHoliday(day: number): boolean {
    return (day % 36 === 0);
  }

  // Tab switching method
  switchToLittlesLawTab(): void {
    this.activeTab = 'littles-law';
    
    // Debug: Check data for each plant
    console.log('🔍 Checking data for each plant in Little\'s Law tab:');
    this.plants.forEach(plant => {
      const plantData = this.trainingData().filter(d => d.plant === plant);
      console.log(`  ${plant}: ${plantData.length} records`);
      if (plantData.length > 0) {
        console.log(`    Sample:`, plantData[0]);
      }
    });
    
    // Use setTimeout to ensure DOM is rendered before initializing charts
    setTimeout(() => this.initializeLittlesLawCharts(), 100);
  }

  // Chart initialization methods
  initializeLittlesLawCharts(): void {
    const data = this.trainingData();
    if (data.length === 0) {
      console.warn('No training data available for charts');
      return;
    }

    console.log('Initializing Little\'s Law charts with', data.length, 'data points');
    
    // Check if Chart.js is available
    const Chart = (window as any).Chart;
    if (typeof Chart === 'undefined') {
      console.error('Chart.js library is not loaded. Charts will not display.');
      console.error('Make sure Chart.js is imported in main.ts and the server is restarted.');
      this.displayDataWithoutCharts(data);
      return;
    } else {
      console.log('✅ Chart.js is available:', Chart);
      console.log('Available chart types:', Chart.defaults);
    }

    this.initializeLittlesLawScatterPlot(data);
    this.initializeComplianceHistogram(data);
  }

  private displayDataWithoutCharts(_data: DataPoint[]): void {
    console.log('Chart.js not available. Showing fallback visualizations.');
    
    // Show fallback elements
    const scatterFallback = document.getElementById('scatterPlotFallback');
    const complianceFallback = document.getElementById('complianceChartFallback');
    
    if (scatterFallback) {
      scatterFallback.style.display = 'block';
    }
    if (complianceFallback) {
      complianceFallback.style.display = 'block';
    }
    
    console.log('Average compliance:', this.getAverageCompliance());
    console.log('Cycle time range:', this.getCycleTimeRange());
  }

  private initializeLittlesLawScatterPlot(data: DataPoint[]): void {
    const canvas = document.getElementById('littlesLawChart') as HTMLCanvasElement;
    console.log('Scatter plot canvas element:', canvas);
    
    if (!canvas) {
      console.error('Canvas element "littlesLawChart" not found in DOM');
      return;
    }
    
    if (typeof (window as any).Chart === 'undefined') {
      console.error('Chart.js not available for scatter plot');
      return;
    }

    if (this.littlesLawChart) {
      this.littlesLawChart.destroy();
    }

    // Prepare data points
    const chartData = data.map(d => ({
      x: d.throughput,
      y: d.wip,
      r: Math.max(3, Math.min(15, d.cycleTime * 2)), // Point size based on cycle time
      compliance: d.littlesLawCompliance || 0,
      plant: d.plant,
      application: d.application,
      cycleTime: d.cycleTime
    }));

    const ctx = canvas.getContext('2d')!;
    this.littlesLawChart = new (window as any).Chart(ctx, {
      type: 'bubble',
      data: {
        datasets: [
          {
            label: 'WIP vs Throughput (size = Cycle Time, color = Compliance)',
            data: chartData,
            backgroundColor: chartData.map(d => 
              `rgba(25, 118, 210, ${Math.max(0.3, d.compliance / 100)})`
            ),
            borderColor: 'rgba(25, 118, 210, 0.8)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Little\'s Law Analysis: WIP vs Throughput'
          },
          tooltip: {
            callbacks: {
              label: (context: any) => {
                const point = context.raw;
                return [
                  `Plant: ${point.plant}`,
                  `Application: ${point.application}`,
                  `WIP: ${point.y}`,
                  `Throughput: ${point.x.toFixed(1)}`,
                  `Cycle Time: ${point.cycleTime.toFixed(2)} days`,
                  `Compliance: ${point.compliance.toFixed(1)}%`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Throughput (units/day)'
            }
          },
          y: {
            title: {
              display: true,
              text: 'WIP (units)'
            }
          }
        }
      }
    });
  }

  private initializeComplianceHistogram(data: DataPoint[]): void {
    const canvas = document.getElementById('complianceChart') as HTMLCanvasElement;
    console.log('Compliance chart canvas element:', canvas);
    
    if (!canvas) {
      console.error('Canvas element "complianceChart" not found in DOM');
      return;
    }
    
    if (typeof (window as any).Chart === 'undefined') {
      console.error('Chart.js not available for compliance chart');
      return;
    }

    if (this.complianceChart) {
      this.complianceChart.destroy();
    }

    // Create compliance bins
    const bins = Array(10).fill(0); // 10% bins (0-10%, 10-20%, ..., 90-100%)
    data.forEach(d => {
      const compliance = d.littlesLawCompliance || 0;
      const binIndex = Math.min(9, Math.floor(compliance / 10));
      bins[binIndex]++;
    });

    const ctx = canvas.getContext('2d')!;
    this.complianceChart = new (window as any).Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
        datasets: [{
          label: 'Number of Data Points',
          data: bins,
          backgroundColor: 'rgba(25, 118, 210, 0.6)',
          borderColor: 'rgba(25, 118, 210, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Little\'s Law Compliance Distribution'
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Compliance Range'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Frequency'
            }
          }
        }
      }
    });
  }

  // Little's Law Analysis Methods
  getAverageCompliance(): number {
    const data = this.trainingData();
    if (data.length === 0) return 0;
    
    const totalCompliance = data.reduce((sum, d) => sum + (d.littlesLawCompliance || 0), 0);
    return totalCompliance / data.length;
  }

  getCycleTimeRange(): string {
    const data = this.trainingData();
    if (data.length === 0) return 'No data';
    
    const cycleTimes = data.map(d => d.cycleTime);
    const min = Math.min(...cycleTimes);
    const max = Math.max(...cycleTimes);
    return `${min.toFixed(2)} - ${max.toFixed(2)} days`;
  }

  getPlantCompliance(plant: string): number {
    const plantData = this.trainingData().filter(d => d.plant === plant);
    console.log(`🏭 Plant ${plant} data count:`, plantData.length);
    
    if (plantData.length === 0) {
      console.warn(`⚠️ No data found for ${plant}`);
      return 0;
    }
    
    const totalCompliance = plantData.reduce((sum, d) => sum + (d.littlesLawCompliance || 0), 0);
    const avgCompliance = totalCompliance / plantData.length;
    console.log(`✅ ${plant} average compliance:`, avgCompliance);
    return avgCompliance;
  }

  getPlantAvgCycleTime(plant: string): number {
    const plantData = this.trainingData().filter(d => d.plant === plant);
    if (plantData.length === 0) return 0;
    
    const totalCycleTime = plantData.reduce((sum, d) => sum + d.cycleTime, 0);
    return totalCycleTime / plantData.length;
  }

  // Fallback visualization methods
  getWIPRange(): string {
    const data = this.trainingData();
    if (data.length === 0) return 'No data';
    
    const wips = data.map(d => d.wip);
    const min = Math.min(...wips);
    const max = Math.max(...wips);
    return `${min} - ${max} units`;
  }

  getThroughputRange(): string {
    const data = this.trainingData();
    if (data.length === 0) return 'No data';
    
    const throughputs = data.map(d => d.throughput);
    const min = Math.min(...throughputs);
    const max = Math.max(...throughputs);
    return `${min.toFixed(1)} - ${max.toFixed(1)} units/day`;
  }

  getComplianceRanges(): { label: string; count: number; percentage: number }[] {
    const data = this.trainingData();
    if (data.length === 0) return [];

    const bins = Array(10).fill(0);
    data.forEach(d => {
      const compliance = d.littlesLawCompliance || 0;
      const binIndex = Math.min(9, Math.floor(compliance / 10));
      bins[binIndex]++;
    });

    const maxCount = Math.max(...bins);
    
    return [
      '0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
      '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
    ].map((label, index) => ({
      label,
      count: bins[index],
      percentage: maxCount > 0 ? (bins[index] / maxCount) * 100 : 0
    }));
  }
}
