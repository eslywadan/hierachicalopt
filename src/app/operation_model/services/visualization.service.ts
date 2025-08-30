import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

export interface TrainingVisualizationData {
  dashboard_created: boolean;
  visualization_path?: string;
  training_metrics?: {
    loss: number[];
    val_loss: number[];
    mae: number[];
    r2_score: number[];
  };
  current_status?: {
    status: string;
    current_epoch: number;
    total_epochs: number;
    current_batch: number;
    total_batches: number;
    elapsed_minutes: number;
    learning_rate: number;
    model_id: string;
  };
  training_logs?: Array<{
    timestamp: string;
    level: string;
    message: string;
  }>;
}

export interface ModelArchitectureData {
  visualization_created: boolean;
  visualization_path?: string;
  model_info?: {
    total_params: number;
    trainable_params: number;
    input_shape: string;
    output_shape: string;
    layers: Array<{
      name: string;
      type: string;
      output_shape: string;
      params: number;
    }>;
  };
}

@Injectable({
  providedIn: 'root'
})
export class VisualizationService {
  private baseUrl = 'http://localhost:5001/api';

  constructor(private http: HttpClient) {}

  /**
   * Get training dashboard visualization data
   */
  getTrainingDashboard(): Observable<TrainingVisualizationData> {
    return this.http.get<TrainingVisualizationData>(`${this.baseUrl}/training/status/dashboard`);
  }

  /**
   * Get model architecture visualization
   */
  getModelArchitecture(modelId: string): Observable<ModelArchitectureData> {
    return this.http.get<ModelArchitectureData>(`${this.baseUrl}/model/${modelId}/visualize`);
  }

  /**
   * Get training logs
   */
  getTrainingLogs(): Observable<Array<{timestamp: string, level: string, message: string}>> {
    return this.http.get<any>(`${this.baseUrl}/training/logs`).pipe(
      map((response: any) => {
        if (response.success && response.logs) {
          return response.logs.map((log: any) => {
            // Safely parse timestamp
            let timestamp: string;
            if (log.timestamp) {
              try {
                const parsedDate = new Date(log.timestamp);
                timestamp = isNaN(parsedDate.getTime()) ? new Date().toISOString() : parsedDate.toISOString();
              } catch {
                timestamp = new Date().toISOString();
              }
            } else {
              timestamp = new Date().toISOString();
            }
            
            return {
              timestamp,
              level: log.level?.toUpperCase() || 'INFO',
              message: log.message || 'No message'
            };
          });
        }
        return [];
      }),
      catchError(() => of([]))
    );
  }

  /**
   * Get visualization image as blob for display
   */
  getVisualizationImage(imagePath: string): Observable<Blob> {
    // Remove 'visualizations/' prefix if present and construct full URL
    const cleanPath = imagePath.replace('visualizations/', '');
    return this.http.get(`http://localhost:5001/visualizations/${cleanPath}`, { 
      responseType: 'blob',
      headers: {
        'Accept': 'image/png'
      }
    });
  }
}