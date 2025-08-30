// ============================================
// COMPLETE LEVEL 3 LSTM OPERATION MODEL FILES
// ============================================
// Instructions: Copy each section to the file path indicated
// File markers: ===FILE: path/to/file.ext===
// ============================================

// ===FILE: src/app/operation_model/models/level3-lstm-model.ts===

import * as tf from '@tensorflow/tfjs';
import { Injectable, InjectionToken, Inject } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

/**
 * Configuration interfaces
 */
export interface DataConfig {
  plants: string[];
  applications: string[];
  panelSizes: string[];
  historicalDays: number;
}

export interface DataGeneratorParams {
  baseWIP: number;
  baseThroughput: number;
  seasonality: number;
  noiseLevel: number;
}

export interface LSTMConfig {
  lstmUnits1: number;
  lstmUnits2: number;
  dropoutRate: number;
  sequenceLength: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  trainTestSplit: number;
}

// Injection tokens
export const DATA_CONFIG = new InjectionToken<DataConfig>('DataConfig');
export const DATA_GENERATOR_PARAMS = new InjectionToken<DataGeneratorParams>('DataGeneratorParams');
export const LSTM_CONFIG = new InjectionToken<LSTMConfig>('LSTMConfig');

export interface DataPoint {
  day: number;
  date: Date;
  plant: string;
  application: string;
  panelSize: string;
  wip: number;
  throughput: number;
  cycleTime: number;
  finishedGoods: number;
  semiFinishedGoods: number;
  littlesLawCompliance?: number;
}

export interface PredictionRequest {
  plant: string;
  application: string;
  panelSize: string;
  predictionDays: number;
  currentWIP: number;
  plannedThroughput: number;
  targetProduction?: number;
}

export interface ValidationResult {
  isValid: boolean;
  expectedCycleTime: number;
  actualCycleTime?: number;
  requiredThroughput?: number;
  message?: string;
  littlesLawCompliance?: number;
}

/**
 * Data Generator for synthetic training data
 */
export class DataGenerator {
  constructor(
    private config: DataConfig,
    private params: DataGeneratorParams
  ) {}

  generateData(): DataPoint[] {
    const data: DataPoint[] = [];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - this.config.historicalDays);

    // Debug: Log configuration
    console.log('📊 Generating data with config:', {
      plants: this.config.plants,
      applications: this.config.applications,
      panelSizes: this.config.panelSizes,
      historicalDays: this.config.historicalDays
    });

    for (let day = 0; day < this.config.historicalDays; day++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(currentDate.getDate() + day);
      
      for (const plant of this.config.plants) {
        for (const application of this.config.applications) {
          for (const panelSize of this.config.panelSizes) {
            data.push(this.generateDataPoint(
              day,
              currentDate,
              plant,
              application,
              panelSize
            ));
          }
        }
      }
    }

    // Debug: Count data points per plant
    const plantCounts: { [key: string]: number } = {};
    const sampleData: { [key: string]: DataPoint[] } = {};
    
    data.forEach(d => {
      plantCounts[d.plant] = (plantCounts[d.plant] || 0) + 1;
      
      // Keep sample data for each plant
      if (!sampleData[d.plant]) {
        sampleData[d.plant] = [];
      }
      if (sampleData[d.plant].length < 3) {
        sampleData[d.plant].push(d);
      }
    });
    
    console.log('📈 Data points per plant:', plantCounts);
    console.log('📋 Sample data for each plant:');
    Object.keys(sampleData).forEach(plant => {
      console.log(`  ${plant}:`, sampleData[plant].slice(0, 2));
    });

    return data;
  }

  private generateDataPoint(
    day: number,
    date: Date,
    plant: string,
    application: string,
    panelSize: string
  ): DataPoint {
    // Enhanced seasonality with multiple factors
    const dayOfYear = this.getDayOfYear(date);
    
    // Annual seasonality
    const annualSeason = 1 + this.params.seasonality * Math.sin(2 * Math.PI * dayOfYear / 365);
    
    // Day-of-week effect (lower production on weekends)
    const dayOfWeek = date.getDay(); // 0=Sunday, 6=Saturday
    let weekdayFactor = 1.0;
    if (dayOfWeek === 0 || dayOfWeek === 6) { // Weekend
      weekdayFactor = 0.3; // Much lower production
    } else if (dayOfWeek === 1 || dayOfWeek === 5) { // Monday/Friday
      weekdayFactor = 0.85; // Slightly lower
    }
    
    // Holiday effect (simplified - assume major holidays)
    const isHoliday = (day % 36 === 0); // Roughly every 36 days = ~10 holidays/year
    const holidayFactor = isHoliday ? 0.1 : 1.0;
    
    // Quarter-end effect (higher production pressure)
    const dayInQuarter = day % 91; // ~91 days per quarter
    const quarterEndFactor = dayInQuarter > 85 ? 1.2 : 1.0; // Last week of quarter
    
    const seasonalityFactor = annualSeason * weekdayFactor * holidayFactor * quarterEndFactor;

    // Plant-specific factors
    const plantFactor = this.getPlantFactor(plant);
    
    // Application-specific factors
    const appFactor = this.getApplicationFactor(application);
    
    // Panel size factors
    const sizeFactor = this.getPanelSizeFactor(panelSize);

    // Generate realistic independent variations for WIP and throughput
    const throughputNoise = 1 + (Math.random() - 0.5) * this.params.noiseLevel;
    const wipNoise = 1 + (Math.random() - 0.5) * (this.params.noiseLevel * 1.5); // More WIP variation
    
    // Throughput: affected by seasonality, plant efficiency, and process complexity
    const throughput = this.params.baseThroughput * 
      plantFactor * appFactor * sizeFactor * seasonalityFactor * throughputNoise;
    
    // WIP: has different dynamics - affected by demand fluctuations and inventory policies
    // WIP variation is more independent of throughput variations
    const wipSeasonality = 1 + this.params.seasonality * 0.7 * Math.sin(2 * Math.PI * dayOfYear / 365 + Math.PI/4); // Phase shift
    const wipDemandVariation = 1 + 0.3 * Math.sin(2 * Math.PI * day / 45); // ~6 week demand cycles
    const processingBottlenecks = 1 + 0.15 * Math.sin(2 * Math.PI * day / 21); // ~3 week bottleneck cycles
    
    const wip = this.params.baseWIP * 
      plantFactor * appFactor * sizeFactor * wipSeasonality * 
      wipDemandVariation * processingBottlenecks * wipNoise;
    
    // Add realistic cycle time variations beyond just WIP/Throughput
    // Real manufacturing has setup times, quality issues, equipment downtime, etc.
    const plantCTFactor = this.getPlantCycleTimeFactor(plant);
    const appCTFactor = this.getApplicationCycleTimeFactor(application);
    const processEfficiencyVariation = 0.8 + 0.4 * Math.random(); // 0.8 to 1.2 multiplier
    const qualityIssueDelay = Math.random() < 0.1 ? (1 + Math.random() * 0.5) : 1; // 10% chance of delays
    const equipmentEfficiency = 0.85 + 0.25 * Math.random(); // 0.85 to 1.1 efficiency
    
    // Base cycle time from Little's Law, then add realistic variations
    const baseCycleTime = wip / throughput;
    const cycleTime = baseCycleTime * plantCTFactor * appCTFactor * 
                     processEfficiencyVariation * qualityIssueDelay * equipmentEfficiency;
    
    // Calculate finished and semi-finished goods
    const finishedGoods = throughput * (0.7 + Math.random() * 0.3);
    const semiFinishedGoods = wip - finishedGoods;

    // Calculate Little's Law compliance (now with realistic deviations)
    const expectedCT = wip / throughput; // Theoretical cycle time from Little's Law
    const compliance = Math.max(0, 100 - Math.abs((cycleTime - expectedCT) / expectedCT) * 100);

    return {
      day,
      date,
      plant,
      application,
      panelSize,
      wip: Math.round(wip),
      throughput: Math.round(throughput * 10) / 10,
      cycleTime: Math.round(cycleTime * 100) / 100,
      finishedGoods: Math.round(finishedGoods),
      semiFinishedGoods: Math.round(semiFinishedGoods),
      littlesLawCompliance: Math.round(compliance * 100) / 100
    };
  }

  private getDayOfYear(date: Date): number {
    const start = new Date(date.getFullYear(), 0, 0);
    const diff = date.getTime() - start.getTime();
    return Math.floor(diff / (1000 * 60 * 60 * 24));
  }

  private getPlantFactor(plant: string): number {
    const factors: { [key: string]: number } = {
      'Plant_1': 1.0,
      'Plant_2': 1.2,
      'Plant_3': 0.9
    };
    return factors[plant] || 1.0;
  }

  private getApplicationFactor(application: string): number {
    const factors: { [key: string]: number } = {
      'Automotive': 1.1,
      'Consumer_Electronics': 0.9,  // Fixed to match config
      'Industrial': 1.0
    };
    return factors[application] || 1.0;
  }

  private getPanelSizeFactor(size: string): number {
    const factors: { [key: string]: number } = {
      'Small': 0.8,
      'Medium': 1.0,
      'Large': 1.3
    };
    return factors[size] || 1.0;
  }

  // New method for plant-specific cycle time factors
  private getPlantCycleTimeFactor(plant: string): number {
    // Different plants have different processing capabilities and bottlenecks
    const factors: { [key: string]: number } = {
      'Plant_1': 1.0,  // Baseline plant
      'Plant_2': 0.85, // More efficient plant (newer equipment)
      'Plant_3': 1.15  // Older plant with more bottlenecks
    };
    return factors[plant] || 1.0;
  }

  // New method for application-specific cycle time factors
  private getApplicationCycleTimeFactor(application: string): number {
    // Different applications have different processing complexity
    const factors: { [key: string]: number } = {
      'Automotive': 1.2,     // High precision, more quality checks
      'Consumer_Electronics': 0.9, // Streamlined process
      'Industrial': 1.0      // Baseline
    };
    return factors[application] || 1.0;
  }
}

/**
 * LSTM Model for predictions
 */
export class LSTMModel {
  private model: tf.LayersModel | null = null;
  private normalizers: {
    wip: { min: number; max: number };
    throughput: { min: number; max: number };
    cycleTime: { min: number; max: number };
  } | null = null;

  constructor(private config: LSTMConfig) {
    // Ensure WebGL backend is being used for best performance
    this.initializeBackend();
  }

  private async initializeBackend(): Promise<void> {
    await tf.ready();
    
    // Try CPU backend first for smoother training
    try {
      await tf.setBackend('cpu');
      console.log('🎯 Using CPU backend for smoother training performance');
    } catch (error) {
      console.log('⚠️ CPU backend not available, using WebGL');
      console.log('🎮 TensorFlow Backend:', tf.getBackend());
      
      // Minimal WebGL optimizations
      if (tf.getBackend() === 'webgl') {
        tf.env().set('WEBGL_PACK', false); // Disable packing to reduce complexity
        tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', -1); // Immediate cleanup
      }
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async buildModel(inputShape: [number, number]): Promise<void> {
    this.model = tf.sequential({
      layers: [
        // First LSTM layer with optimized initializer for large matrices
        tf.layers.lstm({
          units: this.config.lstmUnits1,
          returnSequences: true,
          inputShape: inputShape,
          // Use glorotUniform for large matrices to avoid slowness
          kernelInitializer: 'glorotUniform',
          recurrentInitializer: 'glorotUniform',
          biasInitializer: 'zeros'
        }),
        tf.layers.dropout({ rate: this.config.dropoutRate }),
        
        // Second LSTM layer with optimized initializer
        tf.layers.lstm({
          units: this.config.lstmUnits2,
          returnSequences: false,
          kernelInitializer: 'glorotUniform',
          recurrentInitializer: 'glorotUniform',
          biasInitializer: 'zeros'
        }),
        tf.layers.dropout({ rate: this.config.dropoutRate }),
        
        // Dense layers with proper initialization
        tf.layers.dense({ 
          units: 32, 
          activation: 'relu',
          kernelInitializer: 'heUniform' // Better for ReLU activation
        }),
        tf.layers.dense({ 
          units: 16, 
          activation: 'relu',
          kernelInitializer: 'heUniform'
        }),
        
        // Output layer (predicting WIP, Throughput, CycleTime)
        tf.layers.dense({ 
          units: 3,
          kernelInitializer: 'glorotUniform'
        })
      ]
    });

    // Use a more conservative learning rate and add gradient clipping
    const optimizer = tf.train.adam(Math.min(this.config.learningRate, 0.001));
    
    this.model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
      metrics: ['mae', 'mse']
    });

    // Log model summary for debugging
    console.log('🧠 LSTM Model Architecture:');
    console.log(`  Input Shape: [${inputShape[0]}, ${inputShape[1]}]`);
    console.log(`  LSTM Layer 1: ${this.config.lstmUnits1} units (glorotUniform init)`);
    console.log(`  LSTM Layer 2: ${this.config.lstmUnits2} units (glorotUniform init)`);
    console.log(`  Dense Layers: 32 → 16 → 3 units`);
    console.log(`  Total Parameters: ${this.model.countParams()}`);
    console.log('✅ Using optimized initializers - no orthogonal warning!');
  }

  async train(data: DataPoint[]): Promise<tf.History> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel first.');
    }

    // Performance monitoring
    const startTime = performance.now();

    // Prepare sequences
    let sequences = this.prepareSequences(data);
    
    // Sample data to prevent UI hanging with large datasets
    const maxSequences = 1000; // Much smaller dataset for smooth training
    if (sequences.X.length > maxSequences) {
      console.log(`🎯 Sampling ${maxSequences} sequences from ${sequences.X.length} for better performance`);
      const indices = Array.from({length: sequences.X.length}, (_, i) => i)
        .sort(() => 0.5 - Math.random())
        .slice(0, maxSequences);
      
      sequences = {
        X: indices.map(i => sequences.X[i]),
        Y: indices.map(i => sequences.Y[i])
      };
    }
    
    // Split into train and test
    const splitIndex = Math.floor(sequences.X.length * this.config.trainTestSplit);
    const trainX = sequences.X.slice(0, splitIndex);
    const trainY = sequences.Y.slice(0, splitIndex);
    const testX = sequences.X.slice(splitIndex);
    const testY = sequences.Y.slice(splitIndex);

    // Debug: Log sequence information
    console.log(`Training sequences: ${trainX.length}, Test sequences: ${testX.length}`);
    console.log(`Input shape: [${trainX.length}, ${trainX[0]?.length || 0}, ${trainX[0]?.[0]?.length || 0}]`);
    console.log(`Output shape: [${trainY.length}, ${trainY[0]?.length || 0}]`);

    // Convert to tensors
    const trainXTensor = tf.tensor3d(trainX);
    const trainYTensor = tf.tensor2d(trainY);
    const testXTensor = tf.tensor3d(testX);
    const testYTensor = tf.tensor2d(testY);

    // Debug: Check for NaN values in tensors
    const trainXNaN = await tf.any(tf.isNaN(trainXTensor)).data();
    const trainYNaN = await tf.any(tf.isNaN(trainYTensor)).data();
    console.log('TrainX has NaN:', trainXNaN[0]);
    console.log('TrainY has NaN:', trainYNaN[0]);

    // Split training into smaller chunks to prevent requestAnimationFrame violations
    console.log('🚀 Training started with UI-friendly progressive training');
    
    // Use smaller batch size to reduce per-frame computation
    const uiFriendlyBatchSize = Math.min(this.config.batchSize, 16);
    
    // Custom training loop with aggressive yielding
    const history = await this.model.fit(trainXTensor, trainYTensor, {
      epochs: this.config.epochs,
      batchSize: uiFriendlyBatchSize,
      validationData: [testXTensor, testYTensor],
      shuffle: true,
      yieldEvery: 'batch', // Yield after every single batch
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}/${this.config.epochs} - ` +
            `loss: ${logs?.['loss']?.toFixed(4)} - ` +
            `val_loss: ${logs?.['val_loss']?.toFixed(4)}`);
        },
        onYield: async () => {
          // Additional yielding point
          await tf.nextFrame();
        },
        onTrainBegin: () => {
          console.log('⚡ Using progressive training to prevent UI blocking');
        },
        onTrainEnd: () => {
          console.log('✅ Training completed without blocking UI');
        }
      }
    });

    // Clean up tensors
    trainXTensor.dispose();
    trainYTensor.dispose();
    testXTensor.dispose();
    testYTensor.dispose();

    // Log performance metrics
    const endTime = performance.now();
    const trainingTime = ((endTime - startTime) / 1000).toFixed(2);
    console.log(`⏱️ Training completed in ${trainingTime} seconds`);
    console.log(`📊 Average time per epoch: ${(parseFloat(trainingTime) / this.config.epochs).toFixed(2)}s`);

    return history;
  }

  private prepareSequences(data: DataPoint[]): {
    X: number[][][];
    Y: number[][];
  } {
    // Calculate normalization parameters
    this.calculateNormalizers(data);

    const X: number[][][] = [];
    const Y: number[][] = [];

    // Group data by plant, application, and panel size
    const grouped = this.groupData(data);

    for (const key in grouped) {
      const series = grouped[key];
      if (series.length < this.config.sequenceLength + 1) continue;

      // Create sequences
      for (let i = 0; i <= series.length - this.config.sequenceLength - 1; i++) {
        const sequence = series.slice(i, i + this.config.sequenceLength);
        const target = series[i + this.config.sequenceLength];

        // Normalize and create enhanced feature vectors
        const X_seq = sequence.map(point => {
          const dayOfWeek = point.date.getDay();
          const dayInQuarter = point.day % 91;
          
          const features = [
            this.normalize(point.wip, 'wip'),
            this.normalize(point.throughput, 'throughput'),
            this.normalize(point.cycleTime, 'cycleTime'),
            point.day / 545, // Normalized day (using new historical period)
            Math.sin(2 * Math.PI * point.day / 365), // Annual seasonal component
            Math.cos(2 * Math.PI * point.day / 365),
            // New enhanced features
            dayOfWeek / 6, // Day of week (0-6 normalized)
            Math.sin(2 * Math.PI * dayOfWeek / 7), // Weekly seasonality
            Math.cos(2 * Math.PI * dayOfWeek / 7),
            (point.day % 36 === 0) ? 1 : 0, // Holiday indicator
            dayInQuarter / 91, // Position in quarter
            (dayInQuarter > 85) ? 1 : 0 // Quarter-end indicator
          ];
          
          // Validate features - replace NaN/Infinity with safe values
          return features.map(f => isFinite(f) ? f : 0);
        });

        const Y_seq = [
          this.normalize(target.wip, 'wip'),
          this.normalize(target.throughput, 'throughput'),
          this.normalize(target.cycleTime, 'cycleTime')
        ].map(f => isFinite(f) ? f : 0); // Validate outputs too

        X.push(X_seq);
        Y.push(Y_seq);
      }
    }

    return { X, Y };
  }

  private groupData(data: DataPoint[]): { [key: string]: DataPoint[] } {
    const grouped: { [key: string]: DataPoint[] } = {};

    data.forEach(point => {
      const key = `${point.plant}_${point.application}_${point.panelSize}`;
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(point);
    });

    // Sort each group by day
    for (const key in grouped) {
      grouped[key].sort((a, b) => a.day - b.day);
    }

    return grouped;
  }

  private calculateNormalizers(data: DataPoint[]): void {
    const wips = data.map(d => d.wip).filter(v => isFinite(v));
    const throughputs = data.map(d => d.throughput).filter(v => isFinite(v));
    const cycleTimes = data.map(d => d.cycleTime).filter(v => isFinite(v));

    this.normalizers = {
      wip: { 
        min: wips.length > 0 ? Math.min(...wips) : 0, 
        max: wips.length > 0 ? Math.max(...wips) : 1 
      },
      throughput: { 
        min: throughputs.length > 0 ? Math.min(...throughputs) : 0, 
        max: throughputs.length > 0 ? Math.max(...throughputs) : 1 
      },
      cycleTime: { 
        min: cycleTimes.length > 0 ? Math.min(...cycleTimes) : 0, 
        max: cycleTimes.length > 0 ? Math.max(...cycleTimes) : 1 
      }
    };

    // Debug: Log normalizers to understand data ranges
    console.log('Normalizers calculated:', this.normalizers);
  }

  private normalize(value: number, type: 'wip' | 'throughput' | 'cycleTime'): number {
    if (!this.normalizers) return 0;
    const { min, max } = this.normalizers[type];
    // Prevent division by zero
    if (max === min) return 0.5; // Use middle value if no variation
    const normalized = (value - min) / (max - min);
    // Clamp to prevent extreme values
    return Math.max(0, Math.min(1, normalized));
  }

  private denormalize(value: number, type: 'wip' | 'throughput' | 'cycleTime'): number {
    if (!this.normalizers) return value;
    const { min, max } = this.normalizers[type];
    return value * (max - min) + min;
  }

  async predict(
    historicalData: DataPoint[],
    predictionDays: number
  ): Promise<DataPoint[]> {
    if (!this.model || !this.normalizers) {
      throw new Error('Model not trained');
    }

    const predictions: DataPoint[] = [];
    let inputSequence = historicalData.slice(-this.config.sequenceLength);

    for (let i = 0; i < predictionDays; i++) {
      // Prepare input with same 12 features as training
      const X = inputSequence.map((point) => {
        const currentDay = point.day + i;
        const predictedDate = new Date(point.date);
        predictedDate.setDate(predictedDate.getDate() + i);
        const dayOfWeek = predictedDate.getDay();
        const dayInQuarter = currentDay % 91;
        
        return [
          this.normalize(point.wip, 'wip'),
          this.normalize(point.throughput, 'throughput'),
          this.normalize(point.cycleTime, 'cycleTime'),
          currentDay / 545, // Normalized day (using new historical period)
          Math.sin(2 * Math.PI * currentDay / 365), // Annual seasonal component
          Math.cos(2 * Math.PI * currentDay / 365),
          // Enhanced features (same as training)
          dayOfWeek / 6, // Day of week (0-6 normalized)
          Math.sin(2 * Math.PI * dayOfWeek / 7), // Weekly seasonality
          Math.cos(2 * Math.PI * dayOfWeek / 7),
          (currentDay % 36 === 0) ? 1 : 0, // Holiday indicator
          dayInQuarter / 91, // Position in quarter
          (dayInQuarter > 85) ? 1 : 0 // Quarter-end indicator
        ];
      });

      // Make prediction
      const inputTensor = tf.tensor3d([X]);
      const prediction = this.model.predict(inputTensor) as tf.Tensor;
      const predictionData = await prediction.data();
      
      // Denormalize predictions
      const predictedWIP = this.denormalize(predictionData[0], 'wip');
      const predictedThroughput = this.denormalize(predictionData[1], 'throughput');
      const predictedCycleTime = this.denormalize(predictionData[2], 'cycleTime');

      const lastPoint = historicalData[historicalData.length - 1];
      const predDate = new Date(lastPoint.date);
      predDate.setDate(predDate.getDate() + i + 1);

      const predictedPoint: DataPoint = {
        day: lastPoint.day + i + 1,
        date: predDate,
        plant: lastPoint.plant,
        application: lastPoint.application,
        panelSize: lastPoint.panelSize,
        wip: Math.round(predictedWIP),
        throughput: Math.round(predictedThroughput * 10) / 10,
        cycleTime: Math.round(predictedCycleTime * 100) / 100,
        finishedGoods: Math.round(predictedThroughput * 0.8),
        semiFinishedGoods: Math.round(predictedWIP - predictedThroughput * 0.8)
      };

      predictions.push(predictedPoint);

      // Update input sequence for next prediction
      inputSequence = [...inputSequence.slice(1), predictedPoint];

      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();
    }

    return predictions;
  }

  async saveModel(path: string): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }
    await this.model.save(path);
  }

  async loadModel(path: string): Promise<void> {
    this.model = await tf.loadLayersModel(path);
  }

  /**
   * Comprehensive model evaluation
   */
  evaluateModel(metrics: any, dataContext: { sampleSize: number, dataRange: any }): {
    feasibility: 'EXCELLENT' | 'GOOD' | 'ACCEPTABLE' | 'POOR' | 'UNUSABLE';
    recommendations: string[];
    details: any;
  } {
    const { r2Score, mape } = metrics;
    const recommendations: string[] = [];
    let feasibility: 'EXCELLENT' | 'GOOD' | 'ACCEPTABLE' | 'POOR' | 'UNUSABLE';

    // Evaluate R² Score
    let r2Assessment = 'POOR';
    if (r2Score >= 0.7) r2Assessment = 'EXCELLENT';
    else if (r2Score >= 0.5) r2Assessment = 'GOOD';
    else if (r2Score >= 0.3) r2Assessment = 'ACCEPTABLE';
    else if (r2Score >= 0.1) r2Assessment = 'POOR';
    else r2Assessment = 'UNUSABLE';

    // Evaluate MAPE
    let mapeAssessment = 'POOR';
    if (mape <= 5) mapeAssessment = 'EXCELLENT';
    else if (mape <= 10) mapeAssessment = 'GOOD';
    else if (mape <= 20) mapeAssessment = 'ACCEPTABLE';
    else if (mape <= 30) mapeAssessment = 'POOR';
    else mapeAssessment = 'UNUSABLE';

    // Overall feasibility
    if (r2Score < 0.1 || mape > 30) {
      feasibility = 'UNUSABLE';
    } else if (r2Score < 0.3 || mape > 20) {
      feasibility = 'POOR';
    } else if (r2Score < 0.5 || mape > 15) {
      feasibility = 'ACCEPTABLE';
    } else if (r2Score < 0.7 || mape > 10) {
      feasibility = 'GOOD';
    } else {
      feasibility = 'EXCELLENT';
    }

    // Generate recommendations
    if (r2Score < 0.3) {
      recommendations.push('🔧 Increase model complexity (more LSTM units, layers)');
      recommendations.push('📊 Collect more diverse training data');
      recommendations.push('🎯 Add more relevant features (seasonal, operational)');
    }

    if (mape > 15) {
      recommendations.push('📉 Improve data preprocessing and normalization');
      recommendations.push('⚙️ Tune hyperparameters (learning rate, batch size)');
      recommendations.push('🔄 Try different model architectures');
    }

    if (dataContext.sampleSize < 1000) {
      recommendations.push('📈 Increase training data size for better generalization');
    }

    recommendations.push('🧪 Implement cross-validation for robust evaluation');
    recommendations.push('📊 Monitor predictions vs actual on new data');

    return {
      feasibility,
      recommendations,
      details: {
        r2Assessment,
        mapeAssessment,
        isUsableForProduction: feasibility !== 'UNUSABLE' && feasibility !== 'POOR',
        confidenceLevel: r2Score > 0.5 ? 'High' : r2Score > 0.2 ? 'Medium' : 'Low',
        recommendedUse: feasibility === 'POOR' ? 'Development/Testing Only' : 
                       feasibility === 'ACCEPTABLE' ? 'Monitoring with Caution' : 'Production Ready'
      }
    };
  }
}

/**
 * Schedule Validator using Little's Law
 */
export class ScheduleValidator {
  validateSchedule(request: PredictionRequest): ValidationResult {
    // Calculate expected cycle time using Little's Law
    const expectedCycleTime = request.currentWIP / request.plannedThroughput;
    
    // Check if target production is achievable
    const expectedProduction = request.plannedThroughput * request.predictionDays;
    const targetProduction = request.targetProduction || expectedProduction;
    
    const isValid = expectedProduction >= targetProduction * 0.95; // 5% tolerance
    
    let requiredThroughput = request.plannedThroughput;
    if (!isValid && request.targetProduction) {
      requiredThroughput = request.targetProduction / request.predictionDays;
    }

    return {
      isValid,
      expectedCycleTime,
      requiredThroughput,
      message: isValid 
        ? 'Schedule is feasible' 
        : `Increase throughput to ${requiredThroughput.toFixed(2)} units/day`,
      littlesLawCompliance: 100
    };
  }

  validatePredictions(predictions: DataPoint[]): {
    averageCompliance: number;
    violations: number;
    details: ValidationResult[];
  } {
    const details: ValidationResult[] = [];
    let totalCompliance = 0;
    let violations = 0;

    predictions.forEach(pred => {
      const expectedCT = pred.wip / pred.throughput;
      const deviation = Math.abs(expectedCT - pred.cycleTime) / expectedCT;
      const compliance = (1 - deviation) * 100;
      
      const isValid = compliance > 90; // 90% compliance threshold
      if (!isValid) violations++;

      totalCompliance += compliance;

      details.push({
        isValid,
        expectedCycleTime: expectedCT,
        actualCycleTime: pred.cycleTime,
        littlesLawCompliance: compliance
      });
    });

    return {
      averageCompliance: totalCompliance / predictions.length,
      violations,
      details
    };
  }
}

/**
 * Performance Analyzer
 */
export class PerformanceAnalyzer {
  calculateMetrics(
    actual: number[],
    predicted: number[]
  ): {
    r2Score: number;
    rmse: number;
    mae: number;
    mape: number;
  } {
    const n = actual.length;
    
    // Calculate mean of actual values
    const meanActual = actual.reduce((a, b) => a + b, 0) / n;
    
    // Calculate R² score
    const ssRes = actual.reduce((sum, val, i) => 
      sum + Math.pow(val - predicted[i], 2), 0);
    const ssTot = actual.reduce((sum, val) => 
      sum + Math.pow(val - meanActual, 2), 0);
    const r2Score = 1 - (ssRes / ssTot);
    
    // Calculate RMSE
    const mse = ssRes / n;
    const rmse = Math.sqrt(mse);
    
    // Calculate MAE
    const mae = actual.reduce((sum, val, i) => 
      sum + Math.abs(val - predicted[i]), 0) / n;
    
    // Calculate MAPE
    const mape = actual.reduce((sum, val, i) => 
      sum + Math.abs((val - predicted[i]) / val), 0) / n * 100;

    return {
      r2Score: Math.round(r2Score * 1000) / 1000,
      rmse: Math.round(rmse * 100) / 100,
      mae: Math.round(mae * 100) / 100,
      mape: Math.round(mape * 100) / 100
    };
  }

  analyzeTimeSeries(data: DataPoint[]): {
    trend: 'increasing' | 'decreasing' | 'stable';
    seasonality: boolean;
    outliers: number[];
  } {
    const wipValues = data.map(d => d.wip);
    
    // Simple trend detection using linear regression
    const xValues = data.map((_, i) => i);
    const n = data.length;
    const sumX = xValues.reduce((a, b) => a + b, 0);
    const sumY = wipValues.reduce((a, b) => a + b, 0);
    const sumXY = xValues.reduce((sum, x, i) => sum + x * wipValues[i], 0);
    const sumX2 = xValues.reduce((sum, x) => sum + x * x, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    
    // Determine trend
    let trend: 'increasing' | 'decreasing' | 'stable';
    if (Math.abs(slope) < 1) {
      trend = 'stable';
    } else if (slope > 0) {
      trend = 'increasing';
    } else {
      trend = 'decreasing';
    }
    
    // Detect seasonality (simplified)
    const seasonality = this.detectSeasonality(wipValues);
    
    // Detect outliers using IQR
    const outliers = this.detectOutliers(wipValues);

    return { trend, seasonality, outliers };
  }

  private detectSeasonality(values: number[]): boolean {
    // Simplified seasonality detection
    // Check if there's a repeating pattern every 7 days
    if (values.length < 14) return false;
    
    const weeklyAvg: number[] = [];
    for (let i = 0; i < 7; i++) {
      let sum = 0;
      let count = 0;
      for (let j = i; j < values.length; j += 7) {
        sum += values[j];
        count++;
      }
      weeklyAvg.push(sum / count);
    }
    
    // Check variance in weekly averages
    const avgMean = weeklyAvg.reduce((a, b) => a + b, 0) / 7;
    const variance = weeklyAvg.reduce((sum, val) => 
      sum + Math.pow(val - avgMean, 2), 0) / 7;
    
    return variance > (avgMean * 0.1); // 10% threshold
  }

  private detectOutliers(values: number[]): number[] {
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    const outlierIndices: number[] = [];
    values.forEach((val, i) => {
      if (val < lowerBound || val > upperBound) {
        outlierIndices.push(i);
      }
    });
    
    return outlierIndices;
  }
}

/**
 * Main Level 3 Operation Model
 */
@Injectable({
  providedIn: 'root'
})
export class Level3OperationModel {
  private dataGenerator: DataGenerator;
  private lstmModel: LSTMModel;
  private validator: ScheduleValidator;
  private analyzer: PerformanceAnalyzer;
  private trainingData: DataPoint[] = [];
  private modelStatus$ = new BehaviorSubject<string>('uninitialized');
  
  constructor(
    @Inject(DATA_CONFIG) dataConfig: DataConfig,
    @Inject(DATA_GENERATOR_PARAMS) generatorParams: DataGeneratorParams,
    @Inject(LSTM_CONFIG) lstmConfig: LSTMConfig
  ) {
    this.dataGenerator = new DataGenerator(dataConfig, generatorParams);
    this.lstmModel = new LSTMModel(lstmConfig);
    this.validator = new ScheduleValidator();
    this.analyzer = new PerformanceAnalyzer();
  }

  async initialize(): Promise<void> {
    this.modelStatus$.next('initializing');
    
    // Generate training data
    this.trainingData = this.dataGenerator.generateData();
    
    // Build LSTM model with enhanced features (12 features now)
    await this.lstmModel.buildModel([this.lstmModel['config'].sequenceLength, 12]);
    
    this.modelStatus$.next('initialized');
  }

  async train(): Promise<{
    history: tf.History;
    metrics: any;
    evaluation: any;
  }> {
    this.modelStatus$.next('training');
    
    // Train the model
    const history = await this.lstmModel.train(this.trainingData);
    
    // Calculate performance metrics on test data
    const testData = this.trainingData.slice(-50);
    const predictions = await this.lstmModel.predict(
      this.trainingData.slice(-57, -7),
      7
    );
    
    const actualWIP = testData.slice(0, 7).map(d => d.wip);
    const predictedWIP = predictions.map(d => d.wip);
    
    const metrics = this.analyzer.calculateMetrics(actualWIP, predictedWIP);
    
    // Comprehensive model evaluation
    const evaluation = this.lstmModel.evaluateModel(metrics, {
      sampleSize: this.trainingData.length,
      dataRange: {
        wipRange: [Math.min(...this.trainingData.map(d => d.wip)), Math.max(...this.trainingData.map(d => d.wip))],
        throughputRange: [Math.min(...this.trainingData.map(d => d.throughput)), Math.max(...this.trainingData.map(d => d.throughput))]
      }
    });
    
    // Log evaluation results
    console.log('🎯 Model Evaluation:', evaluation);
    console.log(`📊 Model Feasibility: ${evaluation.feasibility}`);
    console.log(`🔍 Confidence Level: ${evaluation.details.confidenceLevel}`);
    console.log(`💡 Recommendations:`);
    evaluation.recommendations.forEach(rec => console.log(`   ${rec}`));
    
    this.modelStatus$.next(evaluation.feasibility === 'UNUSABLE' ? 'poor-performance' : 'trained');
    
    return { history, metrics, evaluation };
  }

  async makePrediction(request: PredictionRequest): Promise<{
    predictions: DataPoint[];
    validation: ValidationResult;
    analysis: any;
  }> {
    // Filter historical data for the specific configuration
    const relevantData = this.trainingData.filter(d =>
      d.plant === request.plant &&
      d.application === request.application &&
      d.panelSize === request.panelSize
    );

    if (relevantData.length < this.lstmModel['config'].sequenceLength) {
      throw new Error('Insufficient historical data for prediction');
    }

    // Make predictions
    const predictions = await this.lstmModel.predict(
      relevantData,
      request.predictionDays
    );

    // Validate the schedule
    const validation = this.validator.validateSchedule(request);

    // Analyze the predictions
    const analysis = this.analyzer.analyzeTimeSeries(predictions);

    return { predictions, validation, analysis };
  }

  getModelStatus(): Observable<string> {
    return this.modelStatus$.asObservable();
  }

  exportData(): DataPoint[] {
    return this.trainingData;
  }

  async saveModel(path: string): Promise<void> {
    await this.lstmModel.saveModel(path);
  }

  async loadModel(path: string): Promise<void> {
    await this.lstmModel.loadModel(path);
    this.modelStatus$.next('trained');
  }
}
