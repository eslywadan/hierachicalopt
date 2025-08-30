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
      plants: ['Plant_1', 'Plant_2', 'Plant_3'],
      applications: ['Automotive', 'Consumer_Electronics', 'Industrial'],
      panelSizes: ['Small', 'Medium', 'Large'],
      historicalDays: 545
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
