import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError, BehaviorSubject } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

export interface TrainingConfig {
  lstmUnits1: number;
  lstmUnits2: number;
  dropoutRate: number;
  sequenceLength: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  trainTestSplit: number;
}

export interface DataConfig {
  plants: string[];
  applications: string[];
  panelSizes: string[];
  historicalDays: number;
  baseWIP: number;
  baseThroughput: number;
  seasonality: number;
  noiseLevel: number;
}

export interface PredictionRequest {
  model_id: string;
  plant: string;
  application: string;
  panelSize: string;
  predictionDays: number;
  currentWIP: number;
  plannedThroughput: number;
  targetProduction?: number;
}

export interface ValidationRequest {
  wip?: number;
  throughput?: number;
  cycleTime?: number;
  targetProduction?: number;
}

export interface TrainingResult {
  success: boolean;
  model_id: string;
  training_history: {
    loss: number[];
    val_loss: number[];
    epochs: number;
  };
  metrics: {
    rmse: number;
    mae: number;
    r2_score: number;
  };
  training_time: number;
  model_summary: any;
}

export interface PredictionResult {
  success: boolean;
  model_id?: string;
  plant?: string;
  predictions: Array<{
    day: number;
    plant?: string;
    predicted_wip: number;
    predicted_throughput: number;
    predicted_cycle_time: number;
    confidence: number;
  }>;
  validation: any;
  little_law_analysis: any;
}

export interface ValidationResult {
  success: boolean;
  validation: {
    is_valid: boolean;
    compliance_score: number;
    expected_values: any;
    deviations: any;
    recommendations: string[];
    warnings: string[];
    analysis: any;
    overall_score: number;
  };
}

@Injectable({
  providedIn: 'root'
})
export class LSTMBackendService {
  private readonly baseUrl = 'http://localhost:5001/api';  // Changed to 5001 to avoid macOS AirPlay conflict
  private trainingProgress$ = new BehaviorSubject<any>(null);
  
  constructor(private http: HttpClient) {}

  /**
   * Check if the backend service is healthy
   */
  checkHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl.replace('/api', '')}/health`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Generate synthetic training data
   */
  generateTrainingData(config: DataConfig): Observable<any> {
    return this.http.post(`${this.baseUrl}/data/generate`, config).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Train parallel plant-specific LSTM models
   */
  trainPlantModelsParallel(trainingConfig: TrainingConfig, plants?: string[], trainingData?: any[]): Observable<any> {
    const payload = {
      ...trainingConfig,
      plants,
      trainingData
    };

    return this.http.post(`${this.baseUrl}/model/train/parallel/plants`, payload).pipe(
      tap(result => {
        console.log('🏭 Parallel plant training completed:', result);
        this.trainingProgress$.next({
          status: 'completed',
          type: 'parallel_plant',
          result
        });
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Get plant training status
   */
  getPlantTrainingStatus(): Observable<any> {
    return this.http.get(`${this.baseUrl}/model/plants/status`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get summary of all plant models
   */
  getPlantModelsSummary(): Observable<any> {
    return this.http.get(`${this.baseUrl}/model/plants/summary`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get best model for a specific plant
   */
  getBestPlantModel(plant: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/model/plant/${plant}/best`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Make predictions using plant-specific model
   */
  predictWithPlantModel(plant: string, predictionRequest: Omit<PredictionRequest, 'model_id'>): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.baseUrl}/model/plant/${plant}/predict`, predictionRequest).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Train LSTM model (single model)
   */
  trainModel(trainingConfig: TrainingConfig, trainingData?: any[]): Observable<TrainingResult> {
    const payload = {
      ...trainingConfig,
      trainingData
    };

    return this.http.post<TrainingResult>(`${this.baseUrl}/model/train`, payload).pipe(
      tap(result => {
        if (result.success) {
          console.log('🎓 Model training completed:', result.model_id);
          this.trainingProgress$.next({
            status: 'completed',
            model_id: result.model_id,
            metrics: result.metrics
          });
        }
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Make predictions using trained model
   */
  predict(request: PredictionRequest): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.baseUrl}/model/predict`, request).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Validate production parameters using Little's Law
   */
  validateProduction(request: ValidationRequest): Observable<ValidationResult> {
    return this.http.post<ValidationResult>(`${this.baseUrl}/model/validate`, request).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * List all trained models
   */
  listModels(): Observable<any> {
    return this.http.get(`${this.baseUrl}/models`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Delete a trained model
   */
  deleteModel(modelId: string): Observable<any> {
    return this.http.delete(`${this.baseUrl}/model/${modelId}`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get training progress observable
   */
  getTrainingProgress(): Observable<any> {
    return this.trainingProgress$.asObservable();
  }

  /**
   * Update training progress
   */
  updateTrainingProgress(progress: any): void {
    this.trainingProgress$.next(progress);
  }

  /**
   * Get default training configuration
   */
  getDefaultTrainingConfig(): TrainingConfig {
    return {
      lstmUnits1: 64,
      lstmUnits2: 32,
      dropoutRate: 0.2,
      sequenceLength: 10,
      epochs: 50,
      batchSize: 32,
      learningRate: 0.00001,
      trainTestSplit: 0.8
    };
  }

  /**
   * Get default data configuration (synchronized with master data)
   */
  getDefaultDataConfig(): DataConfig {
    return {
      plants: ['Taiwan_Fab1', 'China_Fab1', 'Korea_Fab1'], // Use actual plant names from CSV
      applications: ['Commercial Display', 'Consumer TV', 'Gaming Monitor', 'Laptop Display', 'Professional Monitor'], // From demand data
      panelSizes: ['15.6"', '21.5"', '27"', '32"', '43"', '55"', '65"'], // From demand data
      historicalDays: 120, // Updated to match new default
      baseWIP: 100,
      baseThroughput: 50,
      seasonality: 0.2,
      noiseLevel: 0.1
    };
  }

  /**
   * Create optimized training configuration based on data size
   */
  createOptimizedConfig(dataPoints: number): TrainingConfig {
    // Use updated enhanced defaults
    const config: TrainingConfig = {
      lstmUnits1: 32,
      lstmUnits2: 24,
      dropoutRate: 0.25,
      sequenceLength: 15,
      epochs: 12,
      batchSize: 64,
      learningRate: 0.002,
      trainTestSplit: 0.85
    };
    
    // Adjust parameters based on data size
    if (dataPoints < 1000) {
      config.epochs = 8;
      config.batchSize = 32;
      config.lstmUnits1 = 24;
      config.lstmUnits2 = 16;
    } else if (dataPoints > 10000) {
      config.epochs = 20;
      config.batchSize = 128;
      config.lstmUnits1 = 48;
      config.lstmUnits2 = 32;
    }
    
    return config;
  }

  /**
   * Get enhanced training configuration with updated defaults
   */
  getEnhancedTrainingConfigDefaults(): TrainingConfig {
    return {
      lstmUnits1: 32,
      lstmUnits2: 24,
      dropoutRate: 0.25,
      sequenceLength: 15,
      epochs: 12,
      batchSize: 64,
      learningRate: 0.002,
      trainTestSplit: 0.85
    };
  }

  /**
   * Format metrics for display
   */
  formatMetrics(metrics: any): string {
    if (!metrics) return 'No metrics available';
    
    return `R²: ${metrics.r2_score?.toFixed(4) || 'N/A'}, ` +
           `RMSE: ${metrics.rmse?.toFixed(2) || 'N/A'}, ` +
           `MAE: ${metrics.mae?.toFixed(2) || 'N/A'}`;
  }

  /**
   * Validate prediction request
   */
  validatePredictionRequest(request: PredictionRequest): string[] {
    const errors: string[] = [];
    
    if (!request.model_id) {
      errors.push('Model ID is required');
    }
    
    if (!request.plant) {
      errors.push('Plant selection is required');
    }
    
    if (!request.application) {
      errors.push('Application selection is required');
    }
    
    if (!request.panelSize) {
      errors.push('Panel size selection is required');
    }
    
    if (request.predictionDays <= 0 || request.predictionDays > 365) {
      errors.push('Prediction days must be between 1 and 365');
    }
    
    if (request.currentWIP <= 0) {
      errors.push('Current WIP must be positive');
    }
    
    if (request.plannedThroughput <= 0) {
      errors.push('Planned throughput must be positive');
    }
    
    return errors;
  }

  /**
   * Train enhanced LSTM model with advanced techniques
   */
  trainEnhancedModel(trainingConfig: TrainingConfig, trainingData?: any[]): Observable<TrainingResult> {
    const payload = {
      ...trainingConfig,
      trainingData
    };

    return this.http.post<TrainingResult>(`${this.baseUrl}/model/train/enhanced`, payload).pipe(
      tap(result => {
        if (result.success) {
          console.log('🚀 Enhanced model training completed:', result.model_id);
          this.trainingProgress$.next({
            status: 'completed',
            model_id: result.model_id,
            metrics: result.metrics,
            enhanced: true
          });
        }
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Get enhanced training configuration
   */
  getEnhancedTrainingConfig(): Observable<any> {
    return this.http.get(`${this.baseUrl}/model/train/enhanced/config`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get training logs
   */
  getTrainingLogs(): Observable<any> {
    return this.http.get(`${this.baseUrl}/training/logs`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get training dashboard status
   */
  getTrainingDashboard(): Observable<any> {
    return this.http.get(`${this.baseUrl}/training/status/dashboard`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get real-time training metrics for all active jobs
   */
  getRealtimeTrainingMetrics(): Observable<any> {
    return this.http.get(`${this.baseUrl}/training/metrics/realtime`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get parallel training jobs status
   */
  getParallelTrainingJobs(): Observable<any> {
    return this.http.get(`${this.baseUrl}/training/jobs/parallel`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Start parallel enhanced training across multiple plants
   */
  startParallelEnhancedTraining(trainingConfig: TrainingConfig, plants: string[]): Observable<any> {
    const payload = {
      ...trainingConfig,
      plants,
      enhanced: true,
      parallel: true
    };

    return this.http.post(`${this.baseUrl}/model/train/parallel/enhanced`, payload).pipe(
      tap(result => {
        console.log('🚀🏭 Parallel enhanced training started:', result);
        this.trainingProgress$.next({
          status: 'training',
          type: 'parallel_enhanced',
          jobs: plants.length,
          result
        });
      }),
      catchError(this.handleError)
    );
  }

  /**
   * Get epoch-based metrics for a specific training job
   */
  getJobEpochMetrics(jobId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/training/job/${jobId}/metrics/epochs`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Subscribe to real-time training updates via Server-Sent Events
   */
  subscribeToTrainingUpdates(): Observable<any> {
    // Note: This would require SSE implementation on the backend
    // For now, we'll use polling as a fallback
    return this.http.get(`${this.baseUrl}/training/updates/stream`).pipe(
      catchError(this.handleError)
    );
  }

  private handleError = (error: HttpErrorResponse) => {
    let errorMessage = 'An unknown error occurred';
    
    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Client Error: ${error.error.message}`;
    } else {
      // Server-side error
      if (error.status === 0) {
        errorMessage = 'Cannot connect to the backend service. Please ensure the Flask server is running on http://localhost:5001';
      } else {
        errorMessage = `Server Error: ${error.status} - ${error.error?.error || error.message}`;
      }
    }
    
    console.error('🚨 Backend Service Error:', errorMessage);
    this.trainingProgress$.next({
      status: 'error',
      error: errorMessage
    });
    
    return throwError(() => errorMessage);
  };
}