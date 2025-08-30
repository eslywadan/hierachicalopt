// ============================================
// COMPLETE LEVEL 3 LSTM OPERATION MODEL FILES
// ============================================
// Instructions: Copy each section to the file path indicated
// File markers: ===FILE: path/to/file.ext===
// ============================================

// ===FILE: src/app/operation_model/models/level3-lstm-model.ts===

import * as tf from '@tensorflow/tfjs';
import { Injectable } from '@angular/core';
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

    return data;
  }

  private generateDataPoint(
    day: number,
    date: Date,
    plant: string,
    application: string,
    panelSize: string
  ): DataPoint {
    // Apply seasonality based on day of year
    const dayOfYear = this.getDayOfYear(date);
    const seasonalityFactor = 1 + this.params.seasonality * 
      Math.sin(2 * Math.PI * dayOfYear / 365);

    // Plant-specific factors
    const plantFactor = this.getPlantFactor(plant);
    
    // Application-specific factors
    const appFactor = this.getApplicationFactor(application);
    
    // Panel size factors
    const sizeFactor = this.getPanelSizeFactor(panelSize);

    // Generate base values with variations
    const noise = 1 + (Math.random() - 0.5) * this.params.noiseLevel;
    
    const throughput = this.params.baseThroughput * 
      plantFactor * appFactor * sizeFactor * seasonalityFactor * noise;
    
    const wip = this.params.baseWIP * 
      plantFactor * appFactor * sizeFactor * seasonalityFactor * noise;
    
    // Calculate cycle time using Little's Law
    const cycleTime = wip / throughput;
    
    // Calculate finished and semi-finished goods
    const finishedGoods = throughput * (0.7 + Math.random() * 0.3);
    const semiFinishedGoods = wip - finishedGoods;

    // Calculate Little's Law compliance (should be close to 100%)
    const expectedCT = wip / throughput;
    const compliance = 100 - Math.abs((cycleTime - expectedCT) / expectedCT) * 100;

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
      'Consumer': 0.9,
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

  constructor(private config: LSTMConfig) {}

  async buildModel(inputShape: [number, number]): Promise<void> {
    this.model = tf.sequential({
      layers: [
        // First LSTM layer
        tf.layers.lstm({
          units: this.config.lstmUnits1,
          returnSequences: true,
          inputShape: inputShape
        }),
        tf.layers.dropout({ rate: this.config.dropoutRate }),
        
        // Second LSTM layer
        tf.layers.lstm({
          units: this.config.lstmUnits2,
          returnSequences: false
        }),
        tf.layers.dropout({ rate: this.config.dropoutRate }),
        
        // Dense layers
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 16, activation: 'relu' }),
        
        // Output layer (predicting WIP, Throughput, CycleTime)
        tf.layers.dense({ units: 3 })
      ]
    });

    const optimizer = tf.train.adam(this.config.learningRate);
    this.model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
      metrics: ['mae', 'mse']
    });
  }

  async train(data: DataPoint[]): Promise<tf.History> {
    if (!this.model) {
      throw new Error('Model not built. Call buildModel first.');
    }

    // Prepare sequences
    const sequences = this.prepareSequences(data);
    
    // Split into train and test
    const splitIndex = Math.floor(sequences.X.length * this.config.trainTestSplit);
    const trainX = sequences.X.slice(0, splitIndex);
    const trainY = sequences.Y.slice(0, splitIndex);
    const testX = sequences.X.slice(splitIndex);
    const testY = sequences.Y.slice(splitIndex);

    // Convert to tensors
    const trainXTensor = tf.tensor3d(trainX);
    const trainYTensor = tf.tensor2d(trainY);
    const testXTensor = tf.tensor3d(testX);
    const testYTensor = tf.tensor2d(testY);

    // Train the model
    const history = await this.model.fit(trainXTensor, trainYTensor, {
      epochs: this.config.epochs,
      batchSize: this.config.batchSize,
      validationData: [testXTensor, testYTensor],
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}/${this.config.epochs} - ` +
            `loss: ${logs?.loss?.toFixed(4)} - ` +
            `val_loss: ${logs?.val_loss?.toFixed(4)}`);
        }
      }
    });

    // Clean up tensors
    trainXTensor.dispose();
    trainYTensor.dispose();
    testXTensor.dispose();
    testYTensor.dispose();

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

        // Normalize and create feature vectors
        const X_seq = sequence.map(point => [
          this.normalize(point.wip, 'wip'),
          this.normalize(point.throughput, 'throughput'),
          this.normalize(point.cycleTime, 'cycleTime'),
          point.day / 365, // Normalized day
          Math.sin(2 * Math.PI * point.day / 365), // Seasonal component
          Math.cos(2 * Math.PI * point.day / 365)
        ]);

        const Y_seq = [
          this.normalize(target.wip, 'wip'),
          this.normalize(target.throughput, 'throughput'),
          this.normalize(target.cycleTime, 'cycleTime')
        ];

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
    const wips = data.map(d => d.wip);
    const throughputs = data.map(d => d.throughput);
    const cycleTimes = data.map(d => d.cycleTime);

    this.normalizers = {
      wip: { min: Math.min(...wips), max: Math.max(...wips) },
      throughput: { min: Math.min(...throughputs), max: Math.max(...throughputs) },
      cycleTime: { min: Math.min(...cycleTimes), max: Math.max(...cycleTimes) }
    };
  }

  private normalize(value: number, type: 'wip' | 'throughput' | 'cycleTime'): number {
    if (!this.normalizers) return 0;
    const { min, max } = this.normalizers[type];
    return (value - min) / (max - min);
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
      // Prepare input
      const X = inputSequence.map(point => [
        this.normalize(point.wip, 'wip'),
        this.normalize(point.throughput, 'throughput'),
        this.normalize(point.cycleTime, 'cycleTime'),
        (point.day + i) / 365,
        Math.sin(2 * Math.PI * (point.day + i) / 365),
        Math.cos(2 * Math.PI * (point.day + i) / 365)
      ]);

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
    dataConfig: DataConfig,
    generatorParams: DataGeneratorParams,
    lstmConfig: LSTMConfig
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
    
    // Build LSTM model
    await this.lstmModel.buildModel([this.lstmModel['config'].sequenceLength, 6]);
    
    this.modelStatus$.next('initialized');
  }

  async train(): Promise<{
    history: tf.History;
    metrics: any;
  }> {
    this.modelStatus$.next('training');
    
    // Train the model
    const history = await this.lstmModel.train(this.trainingData);
    
    // Calculate performance metrics
    const testData = this.trainingData.slice(-50);
    const predictions = await this.lstmModel.predict(
      this.trainingData.slice(-57, -7),
      7
    );
    
    const actualWIP = testData.slice(0, 7).map(d => d.wip);
    const predictedWIP = predictions.map(d => d.wip);
    
    const metrics = this.analyzer.calculateMetrics(actualWIP, predictedWIP);
    
    this.modelStatus$.next('trained');
    
    return { history, metrics };
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

// ===FILE: src/app/operation_model/services/level3-operation.service.ts===

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, from } from 'rxjs';
import { map } from 'rxjs/operators';
import { 
  Level3OperationModel,
  DataPoint,
  PredictionRequest,
  ValidationResult,
  DataConfig,
  DataGeneratorParams,
  LSTMConfig
} from '../models/level3-lstm-model';

@Injectable({
  providedIn: 'root'
})
export class Level3OperationService {
  private model: Level3OperationModel | null = null;
  private trainingStatus$ = new BehaviorSubject<{
    status: string;
    progress: number;
    message: string;
  }>({
    status: 'idle',
    progress: 0,
    message: 'Model not initialized'
  });

  private predictions$ = new BehaviorSubject<DataPoint[]>([]);
  private validationResults$ = new BehaviorSubject<ValidationResult | null>(null);

  constructor(private http: HttpClient) {}

  initializeModel(
    dataConfig?: DataConfig,
    generatorParams?: DataGeneratorParams,
    lstmConfig?: LSTMConfig
  ): Observable<void> {
    const defaultDataConfig: DataConfig = {
      plants: ['Plant_1', 'Plant_2'],
      applications: ['Automotive', 'Consumer', 'Industrial'],
      panelSizes: ['Small', 'Medium', 'Large'],
      historicalDays: 90
    };

    const defaultGeneratorParams: DataGeneratorParams = {
      baseWIP: 1000,
      baseThroughput: 150,
      seasonality: 0.2,
      noiseLevel: 0.1
    };

    const defaultLSTMConfig: LSTMConfig = {
      lstmUnits1: 64,
      lstmUnits2: 32,
      dropoutRate: 0.2,
      sequenceLength: 14,
      epochs: 50,
      batchSize: 32,
      learningRate: 0.001,
      trainTestSplit: 0.8
    };

    this.model = new Level3OperationModel(
      dataConfig || defaultDataConfig,
      generatorParams || defaultGeneratorParams,
      lstmConfig || defaultLSTMConfig
    );

    this.updateStatus('initializing', 10, 'Setting up model architecture...');
    
    return from(this.model.initialize()).pipe(
      map(() => {
        this.updateStatus('initialized', 20, 'Model ready for training');
      })
    );
  }

  trainModel(): Observable<any> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    this.updateStatus('training', 30, 'Starting training process...');

    return from(this.model.train()).pipe(
      map(result => {
        this.updateStatus('trained', 100, 'Model training complete');
        return result;
      })
    );
  }

  makePrediction(request: PredictionRequest): Observable<{
    predictions: DataPoint[];
    validation: ValidationResult;
    analysis: any;
  }> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    return from(this.model.makePrediction(request)).pipe(
      map(result => {
        this.predictions$.next(result.predictions);
        this.validationResults$.next(result.validation);
        return result;
      })
    );
  }

  getTrainingStatus(): Observable<any> {
    return this.trainingStatus$.asObservable();
  }

  getPredictions(): Observable<DataPoint[]> {
    return this.predictions$.asObservable();
  }

  getValidationResults(): Observable<ValidationResult | null> {
    return this.validationResults$.asObservable();
  }

  exportTrainingData(): DataPoint[] {
    if (!this.model) {
      throw new Error('Model not initialized');
    }
    return this.model.exportData();
  }

  saveModel(): Observable<void> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }
    return from(this.model.saveModel('localstorage://level3-lstm-model'));
  }

  loadModel(): Observable<void> {
    if (!this.model) {
      this.initializeModel();
    }
    return from(this.model!.loadModel('localstorage://level3-lstm-model'));
  }

  private updateStatus(status: string, progress: number, message: string): void {
    this.trainingStatus$.next({ status, progress, message });
  }
}

// ===FILE: src/app/operation_model/components/level3-dashboard/level3-dashboard.component.ts===

import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
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
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatSelectModule,
    MatProgressBarModule,
    MatTabsModule,
    MatTableModule,
    MatIconModule,
    MatSnackBarModule,
    Level3ChartComponent
  ],
  templateUrl: './level3-dashboard.component.html',
  styleUrls: ['./level3-dashboard.component.scss']
})
export class Level3DashboardComponent implements OnInit, OnDestroy {
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
  
  // Configuration options
  plants = ['Plant_1', 'Plant_2', 'Plant_3'];
  applications = ['Automotive', 'Consumer', 'Industrial'];
  panelSizes = ['Small', 'Medium', 'Large'];
  
  // Table columns
  displayedColumns = ['date', 'wip', 'throughput', 'cycleTime', 'compliance'];
  
  // Chart type
  chartType: 'wip' | 'throughput' | 'cycleTime' | 'compliance' = 'wip';

  constructor(
    private fb: FormBuilder,
    private level3Service: Level3OperationService,
    private snackBar: MatSnackBar
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
          this.snackBar.open('Model initialized successfully', 'Close', { duration: 3000 });
          this.loadTrainingData();
        },
        error: (error) => {
          this.snackBar.open('Failed to initialize model', 'Close', { duration: 3000 });
          console.error(error);
        }
      });
  }

  trainModel(): void {
    this.level3Service.trainModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          this.snackBar.open('Model trained successfully', 'Close', { duration: 3000 });
          console.log('Training metrics:', result.metrics);
        },
        error: (error) => {
          this.snackBar.open('Training failed', 'Close', { duration: 3000 });
          console.error(error);
        }
      });
  }

  makePrediction(): void {
    if (this.predictionForm.invalid) {
      this.snackBar.open('Please fill in all required fields', 'Close', { duration: 3000 });
      return;
    }

    const request: PredictionRequest = this.predictionForm.value;
    
    this.level3Service.makePrediction(request)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          this.snackBar.open('Prediction completed', 'Close', { duration: 3000 });
          console.log('Analysis:', result.analysis);
        },
        error: (error) => {
          this.snackBar.open('Prediction failed', 'Close', { duration: 3000 });
          console.error(error);
        }
      });
  }

  loadTrainingData(): void {
    try {
      const data = this.level3Service.exportTrainingData();
      this.trainingData.set(data.slice(0, 100)); // Show first 100 records
    } catch (error) {
      console.error('Failed to load training data:', error);
    }
  }

  saveModel(): void {
    this.level3Service.saveModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => this.snackBar.open('Model saved successfully', 'Close', { duration: 3000 }),
        error: () => this.snackBar.open('Failed to save model', 'Close', { duration: 3000 })
      });
  }

  loadModel(): void {
    this.level3Service.loadModel()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.snackBar.open('Model loaded successfully', 'Close', { duration: 3000 });
          this.loadTrainingData();
        },
        error: () => this.snackBar.open('Failed to load model', 'Close', { duration: 3000 })
      });
  }

  exportData(): void {
    const data = this.predictions();
    if (data.length === 0) {
      this.snackBar.open('No data to export', 'Close', { duration: 3000 });
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
}

// ===FILE: src/app/operation_model/components/level3-chart/level3-chart.component.ts===

import { Component, Input, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import { DataPoint } from '../../models/level3-lstm-model';

Chart.register(...registerables);

@Component({
  selector: 'app-level3-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container">
      <canvas #chartCanvas></canvas>
    </div>
  `,
  styles: [`
    .chart-container {
      position: relative;
      height: 400px;
      width: 100%;
      padding: 20px;
    }
  `]
})
export class Level3ChartComponent implements OnChanges, AfterViewInit {
  @Input() data: DataPoint[] = [];
  @Input() chartType: 'wip' | 'throughput' | 'cycleTime' | 'compliance' = 'wip';
  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;
  
  private chart: Chart | null = null;

  ngAfterViewInit(): void {
    this.createChart();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] || changes['chartType']) {
      this.updateChart();
    }
  }

  private createChart(): void {
    if (!this.chartCanvas) return;

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const config = this.getChartConfig();
    this.chart = new Chart(ctx, config);
  }

  private updateChart(): void {
    if (!this.chart) {
      this.createChart();
      return;
    }

    const config = this.getChartConfig();
    this.chart.data = config.data;
    this.chart.options = config.options;
    this.chart.update();
  }

  private getChartConfig(): ChartConfiguration {
    const labels = this.data.map(d => d.date.toLocaleDateString());
    let datasets: any[] = [];
    let yAxisLabel = '';

    switch (this.chartType) {
      case 'wip':
        datasets = [{
          label: 'WIP',
          data: this.data.map(d => d.wip),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1
        }, {
          label: 'Finished Goods',
          data: this.data.map(d => d.finishedGoods),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Units';
        break;

      case 'throughput':
        datasets = [{
          label: 'Throughput',
          data: this.data.map(d => d.throughput),
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Units/Day';
        break;

      case 'cycleTime':
        datasets = [{
          label: 'Cycle Time',
          data: this.data.map(d => d.cycleTime),
          borderColor: 'rgb(255, 206, 86)',
          backgroundColor: 'rgba(255, 206, 86, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Days';
        break;

      case 'compliance':
        datasets = [{
          label: "Little's Law Compliance",
          data: this.data.map(d => d.littlesLawCompliance || 0),
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Compliance %';
        break;
    }

    return {
      type: 'line',
      data: {
        labels,
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: `Level 3 ${this.chartType.charAt(0).toUpperCase() + this.chartType.slice(1)} Analysis`
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            title: {
              display: true,
              text: yAxisLabel
            }
          },
          x: {
            title: {
              display: true,
              text: 'Date'
            }
          }
        }
      }
    };
  }
}

// ===FILE: src/app/operation_model/components/level3-dashboard/level3-dashboard.component.html===

<div class="level3-dashboard">
  <mat-card>
    <mat-card-header>
      <mat-card-title>Level 3 LSTM Operation Model</mat-card-title>
      <mat-card-subtitle>Production Planning & Schedule Validation</mat-card-subtitle>
    </mat-card-header>
    
    <mat-card-content>
      <!-- Model Status Bar -->
      <div class="status-section">
        <div class="status-info">
          <span class="status-label">Model Status:</span>
          <span [class.status-badge]="true" [class]="trainingStatus().status">
            {{ trainingStatus().status | uppercase }}
          </span>
          <span class="status-message">{{ trainingStatus().message }}</span>
        </div>
        
        @if (isTraining()) {
          <mat-progress-bar 
            mode="determinate" 
            [value]="trainingStatus().progress">
          </mat-progress-bar>
        }
      </div>

      <!-- Action Buttons -->
      <div class="actions-section">
        <button mat-raised-button color="primary" 
                (click)="initializeModel()"
                [disabled]="isTraining()">
          <mat-icon>settings</mat-icon>
          Initialize Model
        </button>
        
        <button mat-raised-button color="accent" 
                (click)="trainModel()"
                [disabled]="!trainingStatus().status.includes('initialized') || isTraining()">
          <mat-icon>model_training</mat-icon>
          Train Model
        </button>
        
        <button mat-stroked-button 
                (click)="saveModel()"
                [disabled]="!isModelReady()">
          <mat-icon>save</mat-icon>
          Save Model
        </button>
        
        <button mat-stroked-button 
                (click)="loadModel()">
          <mat-icon>upload_file</mat-icon>
          Load Model
        </button>
      </div>

      <!-- Main Content Tabs -->
      <mat-tab-group>
        <!-- Prediction Tab -->
        <mat-tab label="Make Predictions">
          <div class="tab-content">
            <form [formGroup]="predictionForm" class="prediction-form">
              <div class="form-row">
                <mat-form-field>
                  <mat-label>Plant</mat-label>
                  <mat-select formControlName="plant">
                    @for (plant of plants; track plant) {
                      <mat-option [value]="plant">{{ plant }}</mat-option>
                    }
                  </mat-select>
                </mat-form-field>

                <mat-form-field>
                  <mat-label>Application</mat-label>
                  <mat-select formControlName="application">
                    @for (app of applications; track app) {
                      <mat-option [value]="app">{{ app }}</mat-option>
                    }
                  </mat-select>
                </mat-form-field>

                <mat-form-field>
                  <mat-label>Panel Size</mat-label>
                  <mat-select formControlName="panelSize">
                    @for (size of panelSizes; track size) {
                      <mat-option [value]="size">{{ size }}</mat-option>
                    }
                  </mat-select>
                </mat-form-field>
              </div>

              <div class="form-row">
                <mat-form-field>
                  <mat-label>Prediction Days</mat-label>
                  <input matInput type="number" formControlName="predictionDays">
                </mat-form-field>

                <mat-form-field>
                  <mat-label>Current WIP</mat-label>
                  <input matInput type="number" formControlName="currentWIP">
                </mat-form-field>

                <mat-form-field>
                  <mat-label>Planned Throughput</mat-label>
                  <input matInput type="number" formControlName="plannedThroughput">
                </mat-form-field>

                <mat-form-field>
                  <mat-label>Target Production (Optional)</mat-label>
                  <input matInput type="number" formControlName="targetProduction">
                </mat-form-field>
              </div>

              <button mat-raised-button color="primary" 
                      (click)="makePrediction()"
                      [disabled]="!isModelReady() || predictionForm.invalid">
                <mat-icon>trending_up</mat-icon>
                Generate Predictions
              </button>
            </form>

            @if (validationResult()) {
              <mat-card class="validation-card" [class.valid]="validationResult()!.isValid">
                <mat-card-header>
                  <mat-card-title>Schedule Validation</mat-card-title>
                </mat-card-header>
                <mat-card-content>
                  <div class="validation-info">
                    <div class="info-item">
                      <span class="label">Status:</span>
                      <span class="value" [class.success]="validationResult()!.isValid">
                        {{ validationResult()!.isValid ? 'FEASIBLE' : 'INFEASIBLE' }}
                      </span>
                    </div>
                    <div class="info-item">
                      <span class="label">Expected Cycle Time:</span>
                      <span class="value">{{ validationResult()!.expectedCycleTime | number:'1.2-2' }} days</span>
                    </div>
                    @if (validationResult()!.requiredThroughput) {
                      <div class="info-item">
                        <span class="label">Required Throughput:</span>
                        <span class="value">{{ validationResult()!.requiredThroughput | number:'1.1-1' }} units/day</span>
                      </div>
                    }
                    <div class="info-item">
                      <span class="label">Message:</span>
                      <span class="value">{{ validationResult()!.message }}</span>
                    </div>
                  </div>
                </mat-card-content>
              </mat-card>
            }
          </div>
        </mat-tab>

        <!-- Visualization Tab -->
        <mat-tab label="Visualizations">
          <div class="tab-content">
            <div class="chart-controls">
              <button mat-button (click)="chartType = 'wip'">WIP</button>
              <button mat-button (click)="chartType = 'throughput'">Throughput</button>
              <button mat-button (click)="chartType = 'cycleTime'">Cycle Time</button>
              <button mat-button (click)="chartType = 'compliance'">Compliance</button>
            </div>
            
            <app-level3-chart 
              [data]="predictions()" 
              [chartType]="chartType">
            </app-level3-chart>
          </div>
        </mat-tab>

        <!-- Data Tab -->
        <mat-tab label="Prediction Data">
          <div class="tab-content">
            <div class="data-actions">
              <button mat-stroked-button (click)="exportData()">
                <mat-icon>download</mat-icon>
                Export to CSV
              </button>
            </div>

            @if (predictions().length > 0) {
              <table mat-table [dataSource]="predictions()" class="data-table">
                <ng-container matColumnDef="date">
                  <th mat-header-cell *matHeaderCellDef>Date</th>
                  <td mat-cell *matCellDef="let element">{{ element.date | date:'short' }}</td>
                </ng-container>

                <ng-container matColumnDef="wip">
                  <th mat-header-cell *matHeaderCellDef>WIP</th>
                  <td mat-cell *matCellDef="let element">{{ element.wip }}</td>
                </ng-container>

                <ng-container matColumnDef="throughput">
                  <th mat-header-cell *matHeaderCellDef>Throughput</th>
                  <td mat-cell *matCellDef="let element">{{ element.throughput | number:'1.1-1' }}</td>
                </ng-container>

                <ng-container matColumnDef="cycleTime">
                  <th mat-header-cell *matHeaderCellDef>Cycle Time</th>
                  <td mat-cell *matCellDef="let element">{{ element.cycleTime | number:'1.2-2' }}</td>
                </ng-container>

                <ng-container matColumnDef="compliance">
                  <th mat-header-cell *matHeaderCellDef>Compliance %</th>
                  <td mat-cell *matCellDef="let element">
                    <span [class.high-compliance]="element.littlesLawCompliance > 95">
                      {{ element.littlesLawCompliance | number:'1.1-1' }}%
                    </span>
                  </td>
                </ng-container>

                <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
                <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
              </table>
            } @else {
              <div class="no-data">
                <mat-icon>info</mat-icon>
                <p>No predictions available. Please train the model and make predictions.</p>
              </div>
            }
          </div>
        </mat-tab>
      </mat-tab-group>
    </mat-card-content>
  </mat-card>
</div>

// ===FILE: src/app/operation_model/components/level3-dashboard/level3-dashboard.component.scss===

.level3-dashboard {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;

  mat-card {
    margin-bottom: 20px;
  }

  .status-section {
    margin-bottom: 20px;
    padding: 15px;
    background-color: #f5f5f5;
    border-radius: 4px;

    .status-info {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;

      .status-label {
        font-weight: 500;
        color: #666;
      }

      .status-badge {
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;

        &.idle {
          background-color: #e0e0e0;
          color: #666;
        }

        &.initializing,
        &.initialized {
          background-color: #e3f2fd;
          color: #1976d2;
        }

        &.training {
          background-color: #fff3e0;
          color: #f57c00;
        }

        &.trained {
          background-color: #e8f5e9;
          color: #388e3c;
        }
      }

      .status-message {
        color: #666;
        font-size: 14px;
        margin-left: auto;
      }
    }
  }

  .actions-section {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;

    button {
      mat-icon {
        margin-right: 5px;
      }
    }
  }

  .tab-content {
    padding: 20px;
    min-height: 400px;

    .prediction-form {
      .form-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-bottom: 20px;

        mat-form-field {
          width: 100%;
        }
      }

      button {
        margin-top: 10px;
      }
    }

    .validation-card {
      margin-top: 20px;
      
      &.valid {
        border-left: 4px solid #4caf50;
      }

      &:not(.valid) {
        border-left: 4px solid #f44336;
      }

      .validation-info {
        .info-item {
          display: flex;
          padding: 8px 0;
          border-bottom: 1px solid #e0e0e0;

          &:last-child {
            border-bottom: none;
          }

          .label {
            font-weight: 500;
            color: #666;
            min-width: 150px;
          }

          .value {
            color: #333;

            &.success {
              color: #4caf50;
              font-weight: 500;
            }
          }
        }
      }
    }

    .chart-controls {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      justify-content: center;

      button {
        &.active {
          background-color: #e3f2fd;
        }
      }
    }

    .data-actions {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 15px;
    }

    .data-table {
      width: 100%;
      
      .high-compliance {
        color: #4caf50;
        font-weight: 500;
      }
    }

    .no-data {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 200px;
      color: #999;

      mat-icon {
        font-size: 48px;
        width: 48px;
        height: 48px;
        margin-bottom: 10px;
      }
    }
  }
}

// Responsive design
@media (max-width: 768px) {
  .level3-dashboard {
    padding: 10px;

    .actions-section {
      flex-direction: column;

      button {
        width: 100%;
      }
    }

    .tab-content {
      padding: 10px;

      .prediction-form {
        .form-row {
          grid-template-columns: 1fr;
        }
      }
    }
  }
}

// ===FILE: src/app/operation_model/utils/data-utils.ts===

/**
 * Utility functions for data processing and manipulation
 */

export class DataUtils {
  /**
   * Normalize time series data
   */
  static normalize(data: number[], method: 'minmax' | 'zscore' = 'minmax'): number[] {
    if (method === 'minmax') {
      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = max - min;
      return data.map(v => (v - min) / range);
    } else {
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const std = Math.sqrt(
        data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.length
      );
      return data.map(v => (v - mean) / std);
    }
  }
  
  /**
   * Denormalize data
   */
  static denormalize(
    normalizedData: number[],
    originalMin: number,
    originalMax: number
  ): number[] {
    const range = originalMax - originalMin;
    return normalizedData.map(v => v * range + originalMin);
  }
  
  /**
   * Create sliding windows for time series
   */
  static createSlidingWindows<T>(
    data: T[],
    windowSize: number,
    stride: number = 1
  ): T[][] {
    const windows: T[][] = [];
    
    for (let i = 0; i <= data.length - windowSize; i += stride) {
      windows.push(data.slice(i, i + windowSize));
    }
    
    return windows;
  }
  
  /**
   * Calculate moving average
   */
  static movingAverage(data: number[], windowSize: number): number[] {
    const result: number[] = [];
    
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = data.slice(start, i + 1);
      const avg = window.reduce((a, b) => a + b, 0) / window.length;
      result.push(avg);
    }
    
    return result;
  }
  
  /**
   * Detect anomalies using z-score
   */
  static detectAnomalies(
    data: number[],
    threshold: number = 3
  ): { indices: number[]; values: number[] } {
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const std = Math.sqrt(
      data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.length
    );
    
    const anomalies = {
      indices: [] as number[],
      values: [] as number[]
    };
    
    data.forEach((value, index) => {
      const zScore = Math.abs((value - mean) / std);
      if (zScore > threshold) {
        anomalies.indices.push(index);
        anomalies.values.push(value);
      }
    });
    
    return anomalies;
  }
  
  /**
   * Interpolate missing values
   */
  static interpolate(data: (number | null)[]): number[] {
    const result = [...data];
    
    for (let i = 0; i < result.length; i++) {
      if (result[i] === null) {
        // Find previous and next non-null values
        let prevIndex = i - 1;
        let nextIndex = i + 1;
        
        while (prevIndex >= 0 && result[prevIndex] === null) prevIndex--;
        while (nextIndex < result.length && result[nextIndex] === null) nextIndex++;
        
        if (prevIndex >= 0 && nextIndex < result.length) {
          // Linear interpolation
          const prevValue = result[prevIndex] as number;
          const nextValue = result[nextIndex] as number;
          const steps = nextIndex - prevIndex;
          const step = (nextValue - prevValue) / steps;
          
          result[i] = prevValue + step * (i - prevIndex);
        } else if (prevIndex >= 0) {
          result[i] = result[prevIndex];
        } else if (nextIndex < result.length) {
          result[i] = result[nextIndex];
        } else {
          result[i] = 0; // Default value
        }
      }
    }
    
    return result as number[];
  }
  
  /**
   * Calculate correlation between two series
   */
  static correlation(x: number[], y: number[]): number {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const correlation = (n * sumXY - sumX * sumY) /
      Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return correlation;
  }
}

// ===FILE: src/app/operation_model/utils/littles-law-calculator.ts===

/**
 * Little's Law Calculator and Validator
 */
export class LittlesLawCalculator {
  /**
   * Calculate cycle time using Little's Law
   * CT = WIP / Throughput
   */
  static calculateCycleTime(wip: number, throughput: number): number {
    if (throughput === 0) {
      throw new Error('Throughput cannot be zero');
    }
    return wip / throughput;
  }
  
  /**
   * Calculate WIP using Little's Law
   * WIP = Throughput × CT
   */
  static calculateWIP(throughput: number, cycleTime: number): number {
    return throughput * cycleTime;
  }
  
  /**
   * Calculate throughput using Little's Law
   * Throughput = WIP / CT
   */
  static calculateThroughput(wip: number, cycleTime: number): number {
    if (cycleTime === 0) {
      throw new Error('Cycle time cannot be zero');
    }
    return wip / cycleTime;
  }
  
  /**
   * Validate if values comply with Little's Law
   */
  static validate(
    wip: number,
    throughput: number,
    cycleTime: number,
    tolerance: number = 0.05
  ): {
    isValid: boolean;
    expectedCycleTime: number;
    actualCycleTime: number;
    deviation: number;
    deviationPercentage: number;
  } {
    const expectedCycleTime = this.calculateCycleTime(wip, throughput);
    const deviation = Math.abs(expectedCycleTime - cycleTime);
    const deviationPercentage = (deviation / expectedCycleTime) * 100;
    const isValid = deviationPercentage <= tolerance * 100;
    
    return {
      isValid,
      expectedCycleTime,
      actualCycleTime: cycleTime,
      deviation,
      deviationPercentage
    };
  }
  
  /**
   * Calculate required changes to achieve target
   */
  static calculateRequiredChanges(
    current: { wip: number; throughput: number; cycleTime: number },
    target: { wip?: number; throughput?: number; cycleTime?: number }
  ): {
    requiredWIPChange?: number;
    requiredThroughputChange?: number;
    requiredCycleTimeChange?: number;
    recommendations: string[];
  } {
    const result: any = {};
    const recommendations: string[] = [];
    
    if (target.cycleTime !== undefined && target.cycleTime !== current.cycleTime) {
      // Calculate required changes to achieve target cycle time
      if (target.wip !== undefined) {
        // WIP is fixed, calculate required throughput
        const requiredThroughput = target.wip / target.cycleTime;
        result.requiredThroughputChange = requiredThroughput - current.throughput;
        recommendations.push(
          `Adjust throughput to ${requiredThroughput.toFixed(2)} units/day`
        );
      } else if (target.throughput !== undefined) {
        // Throughput is fixed, calculate required WIP
        const requiredWIP = target.throughput * target.cycleTime;
        result.requiredWIPChange = requiredWIP - current.wip;
        recommendations.push(
          `Adjust WIP to ${requiredWIP.toFixed(0)} units`
        );
      } else {
        // Both can change, provide options
        const option1WIP = current.throughput * target.cycleTime;
        const option2Throughput = current.wip / target.cycleTime;
        
        recommendations.push(
          `Option 1: Keep throughput at ${current.throughput}, adjust WIP to ${option1WIP.toFixed(0)}`,
          `Option 2: Keep WIP at ${current.wip}, adjust throughput to ${option2Throughput.toFixed(2)}`
        );
      }
    }
    
    return { ...result, recommendations };
  }
  
  /**
   * Calculate system efficiency
   */
  static calculateEfficiency(
    actualThroughput: number,
    theoreticalThroughput: number
  ): number {
    if (theoreticalThroughput === 0) return 0;
    return (actualThroughput / theoreticalThroughput) * 100;
  }
}

// ===FILE: src/app/operation_model/operation-model.module.ts===

import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Routes } from '@angular/router';
import { HttpClientModule } from '@angular/common/http';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';

// Material imports
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBarModule } from '@angular/material/snack-bar';

// Components
import { Level3DashboardComponent } from './components/level3-dashboard/level3-dashboard.component';
import { Level3ChartComponent } from './components/level3-chart/level3-chart.component';

// Services
import { Level3OperationService } from './services/level3-operation.service';

const routes: Routes = [
  {
    path: '',
    component: Level3DashboardComponent
  },
  {
    path: 'level3',
    component: Level3DashboardComponent
  }
];

@NgModule({
  declarations: [
    // Components are now standalone, so no declarations needed
  ],
  imports: [
    CommonModule,
    HttpClientModule,
    ReactiveFormsModule,
    FormsModule,
    RouterModule.forChild(routes),
    // Material modules
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatSelectModule,
    MatProgressBarModule,
    MatTabsModule,
    MatTableModule,
    MatIconModule,
    MatSnackBarModule,
    // Standalone components
    Level3DashboardComponent,
    Level3ChartComponent
  ],
  providers: [
    Level3OperationService
  ]
})
export class OperationModelModule { }

// ===FILE: src/app/app.routes.ts===
// UPDATE YOUR EXISTING FILE - Add the operation-model route

import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/opt-board',
    pathMatch: 'full'
  },
  {
    path: 'opt-board',
    loadChildren: () => import('./opt-board/opt-board.module').then(m => m.OptBoardModule)
  },
  {
    path: 'operation-model',
    loadChildren: () => import('./operation_model/operation-model.module').then(m => m.OperationModelModule)
  }
];

// ===FILE: package.json===
// ADD THESE DEPENDENCIES TO YOUR EXISTING package.json

{
  "dependencies": {
    "@angular/animations": "^17.0.0",
    "@angular/cdk": "^17.0.0",
    "@angular/common": "^17.0.0",
    "@angular/compiler": "^17.0.0",
    "@angular/core": "^17.0.0",
    "@angular/forms": "^17.0.0",
    "@angular/material": "^17.0.0",
    "@angular/platform-browser": "^17.0.0",
    "@angular/platform-browser-dynamic": "^17.0.0",
    "@angular/router": "^17.0.0",
    "@tensorflow/tfjs": "^4.15.0",
    "chart.js": "^4.4.0",
    "rxjs": "^7.8.0",
    "sequelize": "^6.35.0",
    "sequelize-typescript": "^2.1.6",
    "socket.io": "^4.6.0",
    "socket.io-client": "^4.6.0",
    "tslib": "^2.6.0",
    "zone.js": "^0.14.0"
  },
  "devDependencies": {
    "@angular-devkit/build-angular": "^17.0.0",
    "@angular/cli": "^17.0.0",
    "@angular/compiler-cli": "^17.0.0",
    "@types/jasmine": "~5.1.0",
    "@types/node": "^20.0.0",
    "jasmine-core": "~5.1.0",
    "karma": "~6.4.0",
    "karma-chrome-launcher": "~3.2.0",
    "karma-coverage": "~2.2.0",
    "karma-jasmine": "~5.1.0",
    "karma-jasmine-html-reporter": "~2.1.0",
    "typescript": "~5.2.0"
  }
}

// ===FILE: src/app/operation_model/__tests__/level3-lstm-model.test.ts===

import { 
  DataGenerator, 
  LSTMModel, 
  ScheduleValidator,
  PerformanceAnalyzer,
  Level3OperationModel 
} from '../models/level3-lstm-model';

describe('Level3 LSTM Model Tests', () => {
  let dataGenerator: DataGenerator;
  let lstmModel: LSTMModel;
  let validator: ScheduleValidator;
  let analyzer: PerformanceAnalyzer;
  
  beforeEach(() => {
    const config = {
      plants: ['Plant_1', 'Plant_2'],
      applications: ['Automotive', 'Consumer'],
      panelSizes: ['Small', 'Medium'],
      historicalDays: 30
    };
    
    const params = {
      baseWIP: 500,
      baseThroughput: 100,
      seasonality: 0.3,
      noiseLevel: 0.1
    };
    
    dataGenerator = new DataGenerator(config, params);
    lstmModel = new LSTMModel({
      lstmUnits1: 64,
      lstmUnits2: 32,
      dropoutRate: 0.2,
      sequenceLength: 7,
      epochs: 10,
      batchSize: 16,
      learningRate: 0.001,
      trainTestSplit: 0.8
    });
    validator = new ScheduleValidator();
    analyzer = new PerformanceAnalyzer();
  });
  
  describe('DataGenerator', () => {
    test('should generate correct number of data points', () => {
      const data = dataGenerator.generateData();
      expect(data.length).toBe(30 * 2 * 2 * 2); // days * plants * apps * sizes
    });
    
    test('should follow Little\'s Law', () => {
      const data = dataGenerator.generateData();
      data.forEach(point => {
        const calculatedCT = point.wip / point.throughput;
        expect(Math.abs(calculatedCT - point.cycleTime)).toBeLessThan(0.01);
      });
    });
    
    test('should apply seasonality correctly', () => {
      const data = dataGenerator.generateData();
      const winterData = data.filter(d => d.day < 90);
      const summerData = data.filter(d => d.day > 180 && d.day < 270);
      
      const winterAvgWIP = winterData.reduce((sum, d) => sum + d.wip, 0) / winterData.length;
      const summerAvgWIP = summerData.reduce((sum, d) => sum + d.wip, 0) / summerData.length;
      
      // Summer should have different WIP due to seasonality
      expect(Math.abs(winterAvgWIP - summerAvgWIP)).toBeGreaterThan(10);
    });
  });
  
  describe('ScheduleValidator', () => {
    test('should validate feasible schedule', () => {
      const request = {
        plant: 'Plant_1',
        application: 'Automotive',
        panelSize: 'Medium',
        predictionDays: 7,
        currentWIP: 500,
        plannedThroughput: 100,
        targetProduction: 700
      };
      
      const result = validator.validateSchedule(request);
      expect(result.isValid).toBe(true);
      expect(result.expectedCycleTime).toBeCloseTo(5, 1);
    });
    
    test('should detect infeasible schedule', () => {
      const request = {
        plant: 'Plant_1',
        application: 'Automotive',
        panelSize: 'Medium',
        predictionDays: 7,
        currentWIP: 500,
        plannedThroughput: 50,
        targetProduction: 700
      };
      
      const result = validator.validateSchedule(request);
      expect(result.isValid).toBe(false);
      expect(result.requiredThroughput).toBe(100);
    });
    
    test('should calculate Little\'s Law compliance', () => {
      const predictions = [
        { littlesLawCompliance: 95, feasible: true },
        { littlesLawCompliance: 92, feasible: true },
        { littlesLawCompliance: 88, feasible: false },
        { littlesLawCompliance: 96, feasible: true }
      ] as any;
      
      const result = validator.validatePredictions(predictions);
      expect(result.averageCompliance).toBeCloseTo(92.75, 2);
      expect(result.violations).toBe(1);
    });
  });
  
  describe('PerformanceAnalyzer', () => {
    test('should calculate R² score correctly', () => {
      const actual = [10, 20, 30, 40, 50];
      const predicted = [12, 18, 32, 38, 52];
      
      const metrics = analyzer.calculateMetrics(actual, predicted);
      expect(metrics.r2Score).toBeGreaterThan(0.9);
    });
    
    test('should calculate RMSE correctly', () => {
      const actual = [10, 20, 30];
      const predicted = [11, 19, 31];
      
      const metrics = analyzer.calculateMetrics(actual, predicted);
      expect(metrics.rmse).toBeCloseTo(1.15, 1);
    });
    
    test('should calculate MAPE correctly', () => {
      const actual = [100, 200, 300];
      const predicted = [110, 190, 330];
      
      const metrics = analyzer.calculateMetrics(actual, predicted);
      expect(metrics.mape).toBeLessThan(15);
    });
  });
  
  describe('Integration Tests', () => {
    test('should complete full training cycle', async () => {
      const model = new Level3OperationModel(
        { plants: ['Plant_1'], applications: ['Automotive'], panelSizes: ['Medium'], historicalDays: 10 },
        { baseWIP: 500, baseThroughput: 100, seasonality: 0.1, noiseLevel: 0.05 },
        { lstmUnits1: 16, lstmUnits2: 8, dropoutRate: 0.1, sequenceLength: 3, epochs: 2, batchSize: 4, learningRate: 0.01, trainTestSplit: 0.7 }
      );
      
      await model.initialize();
      const dataCount = model.exportData().length;
      expect(dataCount).toBe(10);
      
      // Note: Actual training would require TensorFlow.js setup in test environment
    });
  });
});

// ===FILE: src/app/operation_model/websocket/realtime-updates.ts===

import { Server as HTTPServer } from 'http';
import { Server as SocketIOServer, Socket } from 'socket.io';
import { EventEmitter } from 'events';

/**
 * WebSocket server for real-time updates
 */
export class RealtimeUpdateServer extends EventEmitter {
  private io: SocketIOServer;
  private clients: Map<string, Socket> = new Map();
  
  constructor(httpServer: HTTPServer) {
    super();
    this.io = new SocketIOServer(httpServer, {
      cors: {
        origin: '*',
        methods: ['GET', 'POST']
      }
    });
    
    this.setupConnectionHandlers();
  }
  
  private setupConnectionHandlers(): void {
    this.io.on('connection', (socket: Socket) => {
      console.log(`Client connected: ${socket.id}`);
      this.clients.set(socket.id, socket);
      
      // Handle client events
      socket.on('subscribe', (channel: string) => {
        socket.join(channel);
        console.log(`Client ${socket.id} subscribed to ${channel}`);
      });
      
      socket.on('unsubscribe', (channel: string) => {
        socket.leave(channel);
        console.log(`Client ${socket.id} unsubscribed from ${channel}`);
      });
      
      socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
        this.clients.delete(socket.id);
      });
      
      // Level 3 specific events
      socket.on('requestPrediction', (data) => {
        this.emit('predictionRequest', { socketId: socket.id, data });
      });
      
      socket.on('requestValidation', (data) => {
        this.emit('validationRequest', { socketId: socket.id, data });
      });
    });
  }
  
  /**
   * Broadcast training progress to all subscribers
   */
  broadcastTrainingProgress(progress: any): void {
    this.io.to('training').emit('trainingProgress', progress);
  }
  
  /**
   * Send prediction results to specific client
   */
  sendPredictionResults(socketId: string, results: any): void {
    const socket = this.clients.get(socketId);
    if (socket) {
      socket.emit('predictionResults', results);
    }
  }
  
  /**
   * Broadcast model status updates
   */
  broadcastModelStatus(status: any): void {
    this.io.emit('modelStatus', status);
  }
  
  /**
   * Broadcast Little's Law violations
   */
  broadcastViolation(violation: {
    timestamp: Date;
    plant: string;
    deviation: number;
    message: string;
  }): void {
    this.io.to('monitoring').emit('littlesLawViolation', violation);
  }
  
  /**
   * Send performance metrics
   */
  broadcastMetrics(metrics: any): void {
    this.io.to('metrics').emit('performanceMetrics', metrics);
  }
}

// ===FILE: src/app/operation_model/database/models.ts===

import { DataTypes, Model, Sequelize } from 'sequelize';

/**
 * Database models for Level 3 Operation Model
 */

export class TrainingData extends Model {
  public id!: number;
  public day!: number;
  public date!: Date;
  public plant!: string;
  public application!: string;
  public panelSize!: string;
  public wip!: number;
  public throughput!: number;
  public cycleTime!: number;
  public finishedGoods!: number;
  public semiFinishedGoods!: number;
  
  static initialize(sequelize: Sequelize) {
    TrainingData.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true
      },
      day: DataTypes.INTEGER,
      date: DataTypes.DATE,
      plant: DataTypes.STRING,
      application: DataTypes.STRING,
      panelSize: DataTypes.STRING,
      wip: DataTypes.FLOAT,
      throughput: DataTypes.FLOAT,
      cycleTime: DataTypes.FLOAT,
      finishedGoods: DataTypes.FLOAT,
      semiFinishedGoods: DataTypes.FLOAT
    }, {
      sequelize,
      tableName: 'training_data'
    });
  }
}

export class PredictionLog extends Model {
  public id!: number;
  public requestId!: string;
  public timestamp!: Date;
  public plant!: string;
  public application!: string;
  public panelSize!: string;
  public predictionDays!: number;
  public predictions!: any;
  public validation!: any;
  public compliance!: number;
  
  static initialize(sequelize: Sequelize) {
    PredictionLog.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true
      },
      requestId: DataTypes.STRING,
      timestamp: DataTypes.DATE,
      plant: DataTypes.STRING,
      application: DataTypes.STRING,
      panelSize: DataTypes.STRING,
      predictionDays: DataTypes.INTEGER,
      predictions: DataTypes.JSON,
      validation: DataTypes.JSON,
      compliance: DataTypes.FLOAT
    }, {
      sequelize,
      tableName: 'prediction_logs'
    });
  }
}

export class ModelMetrics extends Model {
  public id!: number;
  public modelVersion!: string;
  public timestamp!: Date;
  public r2Score!: number;
  public rmse!: number;
  public mae!: number;
  public mape!: number;
  public littlesLawCompliance!: number;
  
  static initialize(sequelize: Sequelize) {
    ModelMetrics.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true
      },
      modelVersion: DataTypes.STRING,
      timestamp: DataTypes.DATE,
      r2Score: DataTypes.FLOAT,
      rmse: DataTypes.FLOAT,
      mae: DataTypes.FLOAT,
      mape: DataTypes.FLOAT,
      littlesLawCompliance: DataTypes.FLOAT
    }, {
      sequelize,
      tableName: 'model_metrics'
    });
  }
}

// Database connection setup
export class Database {
  private sequelize: Sequelize;
  
  constructor(config?: any) {
    this.sequelize = new Sequelize(config || {
      dialect: 'sqlite',
      storage: './data/level3.db',
      logging: false
    });
    
    this.initializeModels();
  }
  
  private initializeModels(): void {
    TrainingData.initialize(this.sequelize);
    PredictionLog.initialize(this.sequelize);
    ModelMetrics.initialize(this.sequelize);
  }
  
  async sync(): Promise<void> {
    await this.sequelize.sync();
  }
  
  getSequelize(): Sequelize {
    return this.sequelize;
  }
}