import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';

export interface MasterDataConfig {
  plants: string[];
  applications: string[];
  panel_sizes: string[];
  regions: string[];
}

export interface PlantMaster {
  id: string;
  name: string;
  display_name: string;
  location: string;
  region: string;
  capacity: number;
  specializations: string[];
  panel_sizes: string[];
}

export interface ProductMaster {
  id: string;
  name: string;
  display_name: string;
  category: string;
  panel_sizes: string[];
}

export interface PanelSizeMaster {
  id: string;
  display: string;
  diagonal: number;
  category: string;
  applications: string[];
}

export interface MasterDataSummary {
  plants: {
    count: number;
    regions: string[];
    items: PlantMaster[];
  };
  products: {
    count: number;
    categories: string[];
    items: ProductMaster[];
  };
  panel_sizes: {
    count: number;
    categories: string[];
    items: PanelSizeMaster[];
  };
}

export interface DataConsistencyReport {
  timestamp: string;
  total_issues: number;
  is_consistent: boolean;
  plant_issues: string[];
  product_issues: string[];
  panel_size_issues: string[];
  demand_data_issues: string[];
  plant_data_issues: string[];
  recommendations: string[];
}

@Injectable({
  providedIn: 'root'
})
export class MasterDataService {
  private readonly baseUrl = 'http://localhost:5001/api/masterdata';

  constructor(private http: HttpClient) {}

  /**
   * Get synchronized master data configuration
   */
  getMasterDataConfig(): Observable<{success: boolean; config: MasterDataConfig; timestamp: string}> {
    return this.http.get<any>(`${this.baseUrl}/config`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get master data summary for dashboard
   */
  getMasterDataSummary(): Observable<{success: boolean; summary: MasterDataSummary; timestamp: string}> {
    return this.http.get<any>(`${this.baseUrl}/summary`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Validate data consistency across all sources
   */
  validateDataConsistency(): Observable<{success: boolean; report: DataConsistencyReport}> {
    return this.http.post<any>(`${this.baseUrl}/validate`, {}).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Export master data configuration
   */
  exportMasterData(): Observable<{success: boolean; export_data: string}> {
    return this.http.get<any>(`${this.baseUrl}/export`).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Reload master data from CSV files
   */
  reloadMasterData(): Observable<{success: boolean; message: string; config: MasterDataConfig; timestamp: string}> {
    return this.http.post<any>(`${this.baseUrl}/reload`, {}).pipe(
      catchError(this.handleError)
    );
  }

  /**
   * Get issue severity level for UI styling
   */
  getIssueSeverity(issueCount: number): 'success' | 'warning' | 'error' {
    if (issueCount === 0) return 'success';
    if (issueCount <= 3) return 'warning';
    return 'error';
  }

  /**
   * Get consistency score as percentage
   */
  getConsistencyScore(report: DataConsistencyReport): number {
    const totalPossibleIssues = 20; // Rough estimate of total checks
    const score = Math.max(0, ((totalPossibleIssues - report.total_issues) / totalPossibleIssues) * 100);
    return Math.round(score);
  }

  private handleError = (error: any): Observable<never> => {
    console.error('Master Data Service Error:', error);
    throw error;
  };
}