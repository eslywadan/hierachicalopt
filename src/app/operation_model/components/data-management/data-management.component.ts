import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, takeUntil } from 'rxjs';
import { MasterDataService, MasterDataConfig, MasterDataSummary, DataConsistencyReport } from '../../services/master-data.service';

@Component({
  selector: 'app-data-management',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="data-management-container">
      <!-- Header -->
      <div class="header-section">
        <h1>🗃️ Master Data Management</h1>
        <p class="subtitle">Synchronize and validate master data across the hierarchical optimization system</p>
        
        <!-- Status Overview -->
        <div class="status-overview">
          <div class="status-card" [class]="'status-' + getOverallStatus()">
            <div class="status-icon">{{ getStatusIcon() }}</div>
            <div class="status-info">
              <h3>Data Consistency</h3>
              <p>{{ consistencyReport()?.is_consistent ? 'All systems synchronized' : consistencyReport()?.total_issues + ' issues found' }}</p>
              <div class="score-bar">
                <div class="score-fill" [style.width.%]="getConsistencyScore()"></div>
              </div>
              <span class="score-text">{{ getConsistencyScore() }}% consistent</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="actions-section">
        <button (click)="validateConsistency()" [disabled]="isValidating()" class="btn btn-primary">
          {{ isValidating() ? '🔍 Validating...' : '🔍 Validate Data Consistency' }}
        </button>
        <button (click)="reloadMasterData()" [disabled]="isReloading()" class="btn btn-secondary">
          {{ isReloading() ? '🔄 Reloading...' : '🔄 Reload Master Data' }}
        </button>
        <button (click)="exportMasterData()" class="btn btn-info">
          📤 Export Configuration
        </button>
      </div>

      <!-- Master Data Summary -->
      @if (masterDataSummary()) {
        <div class="summary-section">
          <h2>📊 Master Data Summary</h2>
          
          <div class="summary-grid">
            <!-- Plants Summary -->
            <div class="summary-card">
              <h3>🏭 Plants</h3>
              <div class="summary-stats">
                <div class="stat">
                  <span class="stat-value">{{ masterDataSummary()!.plants.count }}</span>
                  <span class="stat-label">Total Plants</span>
                </div>
                <div class="stat">
                  <span class="stat-value">{{ masterDataSummary()!.plants.regions.length }}</span>
                  <span class="stat-label">Regions</span>
                </div>
              </div>
              <div class="summary-items">
                <h4>Plant Details:</h4>
                @for (plant of masterDataSummary()!.plants.items.slice(0, 5); track plant.id) {
                  <div class="item">
                    <strong>{{ plant.display_name }}</strong>
                    <span class="item-meta">{{ plant.location }} • {{ plant.capacity }} units/day</span>
                  </div>
                }
                @if (masterDataSummary()!.plants.items.length > 5) {
                  <div class="item-more">... and {{ masterDataSummary()!.plants.items.length - 5 }} more</div>
                }
              </div>
            </div>

            <!-- Products Summary -->
            <div class="summary-card">
              <h3>📦 Products</h3>
              <div class="summary-stats">
                <div class="stat">
                  <span class="stat-value">{{ masterDataSummary()!.products.count }}</span>
                  <span class="stat-label">Total Products</span>
                </div>
                <div class="stat">
                  <span class="stat-value">{{ masterDataSummary()!.products.categories.length }}</span>
                  <span class="stat-label">Categories</span>
                </div>
              </div>
              <div class="summary-items">
                <h4>Product Categories:</h4>
                @for (category of masterDataSummary()!.products.categories; track category) {
                  <div class="item">
                    <span class="category-tag">{{ category }}</span>
                  </div>
                }
              </div>
            </div>

            <!-- Panel Sizes Summary -->
            <div class="summary-card">
              <h3>📏 Panel Sizes</h3>
              <div class="summary-stats">
                <div class="stat">
                  <span class="stat-value">{{ masterDataSummary()!.panel_sizes.count }}</span>
                  <span class="stat-label">Total Sizes</span>
                </div>
                <div class="stat">
                  <span class="stat-value">{{ masterDataSummary()!.panel_sizes.categories.length }}</span>
                  <span class="stat-label">Size Categories</span>
                </div>
              </div>
              <div class="summary-items">
                <h4>Available Sizes:</h4>
                @for (size of masterDataSummary()!.panel_sizes.items.slice(0, 6); track size.id) {
                  <div class="item">
                    <strong>{{ size.display }}</strong>
                    <span class="item-meta">{{ size.category }}</span>
                  </div>
                }
              </div>
            </div>
          </div>
        </div>
      }

      <!-- Consistency Report -->
      @if (consistencyReport()) {
        <div class="consistency-section">
          <h2>🔍 Data Consistency Report</h2>
          <div class="report-timestamp">
            Generated: {{ consistencyReport()!.timestamp | date:'medium' }}
          </div>

          <!-- Issues Summary -->
          <div class="issues-grid">
            <div class="issue-card" [class.has-issues]="consistencyReport()!.plant_issues.length > 0">
              <h4>🏭 Plant Issues</h4>
              <div class="issue-count">{{ consistencyReport()!.plant_issues.length }}</div>
              @if (consistencyReport()!.plant_issues.length > 0) {
                <ul class="issue-list">
                  @for (issue of consistencyReport()!.plant_issues.slice(0, 3); track $index) {
                    <li>{{ issue }}</li>
                  }
                  @if (consistencyReport()!.plant_issues.length > 3) {
                    <li class="issue-more">... {{ consistencyReport()!.plant_issues.length - 3 }} more issues</li>
                  }
                </ul>
              }
            </div>

            <div class="issue-card" [class.has-issues]="consistencyReport()!.product_issues.length > 0">
              <h4>📦 Product Issues</h4>
              <div class="issue-count">{{ consistencyReport()!.product_issues.length }}</div>
              @if (consistencyReport()!.product_issues.length > 0) {
                <ul class="issue-list">
                  @for (issue of consistencyReport()!.product_issues.slice(0, 3); track $index) {
                    <li>{{ issue }}</li>
                  }
                  @if (consistencyReport()!.product_issues.length > 3) {
                    <li class="issue-more">... {{ consistencyReport()!.product_issues.length - 3 }} more issues</li>
                  }
                </ul>
              }
            </div>

            <div class="issue-card" [class.has-issues]="consistencyReport()!.panel_size_issues.length > 0">
              <h4>📏 Panel Size Issues</h4>
              <div class="issue-count">{{ consistencyReport()!.panel_size_issues.length }}</div>
              @if (consistencyReport()!.panel_size_issues.length > 0) {
                <ul class="issue-list">
                  @for (issue of consistencyReport()!.panel_size_issues.slice(0, 3); track $index) {
                    <li>{{ issue }}</li>
                  }
                  @if (consistencyReport()!.panel_size_issues.length > 3) {
                    <li class="issue-more">... {{ consistencyReport()!.panel_size_issues.length - 3 }} more issues</li>
                  }
                </ul>
              }
            </div>

            <div class="issue-card" [class.has-issues]="consistencyReport()!.demand_data_issues.length > 0">
              <h4>📈 Demand Data Issues</h4>
              <div class="issue-count">{{ consistencyReport()!.demand_data_issues.length }}</div>
              @if (consistencyReport()!.demand_data_issues.length > 0) {
                <ul class="issue-list">
                  @for (issue of consistencyReport()!.demand_data_issues.slice(0, 3); track $index) {
                    <li>{{ issue }}</li>
                  }
                  @if (consistencyReport()!.demand_data_issues.length > 3) {
                    <li class="issue-more">... {{ consistencyReport()!.demand_data_issues.length - 3 }} more issues</li>
                  }
                </ul>
              }
            </div>
          </div>

          <!-- Recommendations -->
          @if (consistencyReport()!.recommendations.length > 0) {
            <div class="recommendations-section">
              <h3>💡 Recommendations</h3>
              <ul class="recommendations-list">
                @for (recommendation of consistencyReport()!.recommendations; track $index) {
                  <li>{{ recommendation }}</li>
                }
              </ul>
            </div>
          }
        </div>
      }

      <!-- Configuration Display -->
      @if (masterDataConfig()) {
        <div class="config-section">
          <h2>⚙️ Current Configuration</h2>
          <div class="config-grid">
            <div class="config-item">
              <h4>Plants</h4>
              <div class="config-tags">
                @for (plant of masterDataConfig()!.plants; track plant) {
                  <span class="tag">{{ plant }}</span>
                }
              </div>
            </div>
            <div class="config-item">
              <h4>Applications</h4>
              <div class="config-tags">
                @for (app of masterDataConfig()!.applications; track app) {
                  <span class="tag">{{ app }}</span>
                }
              </div>
            </div>
            <div class="config-item">
              <h4>Panel Sizes</h4>
              <div class="config-tags">
                @for (size of masterDataConfig()!.panel_sizes; track size) {
                  <span class="tag">{{ size }}</span>
                }
              </div>
            </div>
            <div class="config-item">
              <h4>Regions</h4>
              <div class="config-tags">
                @for (region of masterDataConfig()!.regions; track region) {
                  <span class="tag">{{ region }}</span>
                }
              </div>
            </div>
          </div>
        </div>
      }

      <!-- Activity Log -->
      @if (activityLog().length > 0) {
        <div class="log-section">
          <h2>📋 Activity Log</h2>
          <div class="log-container">
            @for (entry of activityLog().slice(-10); track $index) {
              <div class="log-entry" [class]="'log-' + entry.level">
                <span class="log-time">{{ entry.timestamp | date:'HH:mm:ss' }}</span>
                <span class="log-message">{{ entry.message }}</span>
              </div>
            }
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .data-management-container {
      padding: 2rem;
      max-width: 1400px;
      margin: 0 auto;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .header-section {
      text-align: center;
      margin-bottom: 2rem;
      padding: 2rem;
      background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
      color: white;
      border-radius: 12px;
    }

    .header-section h1 {
      margin: 0 0 0.5rem 0;
      font-size: 2.5rem;
    }

    .subtitle {
      margin: 0 0 2rem 0;
      opacity: 0.9;
      font-size: 1.1rem;
    }

    .status-overview {
      display: flex;
      justify-content: center;
    }

    .status-card {
      display: flex;
      align-items: center;
      gap: 1rem;
      background: rgba(255,255,255,0.1);
      padding: 1.5rem;
      border-radius: 12px;
      min-width: 300px;
    }

    .status-success { border-left: 4px solid #10b981; }
    .status-warning { border-left: 4px solid #f59e0b; }
    .status-error { border-left: 4px solid #ef4444; }

    .status-icon {
      font-size: 2rem;
    }

    .status-info h3 {
      margin: 0 0 0.5rem 0;
      font-size: 1.2rem;
    }

    .status-info p {
      margin: 0 0 1rem 0;
      opacity: 0.9;
    }

    .score-bar {
      width: 200px;
      height: 8px;
      background: rgba(255,255,255,0.3);
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 0.5rem;
    }

    .score-fill {
      height: 100%;
      background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
      transition: width 0.3s ease;
    }

    .score-text {
      font-size: 0.9rem;
      font-weight: 600;
    }

    .actions-section {
      display: flex;
      gap: 1rem;
      justify-content: center;
      margin-bottom: 3rem;
      flex-wrap: wrap;
    }

    .btn {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .btn-primary { background: #4f46e5; color: white; }
    .btn-primary:hover:not(:disabled) { background: #4338ca; }

    .btn-secondary { background: #6b7280; color: white; }
    .btn-secondary:hover:not(:disabled) { background: #4b5563; }

    .btn-info { background: #0ea5e9; color: white; }
    .btn-info:hover:not(:disabled) { background: #0284c7; }

    .summary-section, .consistency-section, .config-section, .log-section {
      background: white;
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 2rem;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .summary-section h2, .consistency-section h2, .config-section h2, .log-section h2 {
      margin: 0 0 1.5rem 0;
      color: #374151;
      border-bottom: 2px solid #e5e7eb;
      padding-bottom: 0.5rem;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 1.5rem;
    }

    .summary-card {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1.5rem;
      background: #f9fafb;
    }

    .summary-card h3 {
      margin: 0 0 1rem 0;
      color: #1f2937;
    }

    .summary-stats {
      display: flex;
      gap: 2rem;
      margin-bottom: 1.5rem;
    }

    .stat {
      text-align: center;
    }

    .stat-value {
      display: block;
      font-size: 2rem;
      font-weight: 700;
      color: #4f46e5;
    }

    .stat-label {
      font-size: 0.875rem;
      color: #6b7280;
    }

    .summary-items h4 {
      margin: 0 0 1rem 0;
      color: #374151;
      font-size: 0.95rem;
    }

    .item {
      margin-bottom: 0.5rem;
      padding: 0.5rem;
      background: white;
      border-radius: 4px;
      border-left: 3px solid #4f46e5;
    }

    .item strong {
      display: block;
      color: #1f2937;
    }

    .item-meta {
      font-size: 0.875rem;
      color: #6b7280;
    }

    .item-more {
      font-style: italic;
      color: #6b7280;
    }

    .category-tag {
      background: #ddd6fe;
      color: #5b21b6;
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .report-timestamp {
      text-align: right;
      color: #6b7280;
      font-size: 0.875rem;
      margin-bottom: 1.5rem;
    }

    .issues-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }

    .issue-card {
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 1.5rem;
      background: #f9fafb;
    }

    .issue-card.has-issues {
      border-left: 4px solid #ef4444;
      background: #fef2f2;
    }

    .issue-card h4 {
      margin: 0 0 1rem 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .issue-count {
      background: #e5e7eb;
      color: #374151;
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-weight: 600;
      font-size: 0.875rem;
    }

    .issue-card.has-issues .issue-count {
      background: #ef4444;
      color: white;
    }

    .issue-list {
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .issue-list li {
      padding: 0.5rem 0;
      border-bottom: 1px solid #f3f4f6;
      font-size: 0.9rem;
      color: #374151;
    }

    .issue-more {
      font-style: italic;
      color: #6b7280;
    }

    .recommendations-section {
      background: #f0fdf4;
      padding: 1.5rem;
      border-radius: 8px;
      border-left: 4px solid #10b981;
    }

    .recommendations-section h3 {
      margin: 0 0 1rem 0;
      color: #166534;
    }

    .recommendations-list {
      margin: 0;
      padding-left: 1.5rem;
    }

    .recommendations-list li {
      margin-bottom: 0.5rem;
      color: #166534;
    }

    .config-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
    }

    .config-item h4 {
      margin: 0 0 1rem 0;
      color: #374151;
    }

    .config-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .tag {
      background: #e0e7ff;
      color: #3730a3;
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      font-size: 0.875rem;
      font-weight: 500;
    }

    .log-container {
      max-height: 300px;
      overflow-y: auto;
      background: #1f2937;
      border-radius: 6px;
      padding: 1rem;
    }

    .log-entry {
      margin-bottom: 0.5rem;
      font-family: monospace;
      font-size: 0.875rem;
      color: #d1d5db;
    }

    .log-time {
      color: #9ca3af;
      margin-right: 0.5rem;
    }

    .log-success .log-message { color: #34d399; }
    .log-warning .log-message { color: #fbbf24; }
    .log-error .log-message { color: #f87171; }
    .log-info .log-message { color: #60a5fa; }
  `]
})
export class DataManagementComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  // Signals
  masterDataConfig = signal<MasterDataConfig | null>(null);
  masterDataSummary = signal<MasterDataSummary | null>(null);
  consistencyReport = signal<DataConsistencyReport | null>(null);
  isValidating = signal(false);
  isReloading = signal(false);
  activityLog = signal<Array<{timestamp: Date, level: string, message: string}>>([]);

  // Computed
  getConsistencyScore = computed(() => {
    const report = this.consistencyReport();
    if (!report) return 0;
    return this.masterDataService.getConsistencyScore(report);
  });

  constructor(private masterDataService: MasterDataService) {}

  ngOnInit() {
    this.loadMasterDataSummary();
    this.loadMasterDataConfig();
    this.addLog('info', 'Data Management dashboard initialized');
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadMasterDataConfig() {
    this.masterDataService.getMasterDataConfig()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.masterDataConfig.set(response.config);
          this.addLog('success', 'Master data configuration loaded');
        },
        error: (error) => {
          this.addLog('error', `Failed to load configuration: ${error.message || error}`);
        }
      });
  }

  loadMasterDataSummary() {
    this.masterDataService.getMasterDataSummary()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.masterDataSummary.set(response.summary);
          this.addLog('success', `Master data summary loaded: ${response.summary.plants.count} plants, ${response.summary.products.count} products`);
        },
        error: (error) => {
          this.addLog('error', `Failed to load summary: ${error.message || error}`);
        }
      });
  }

  validateConsistency() {
    this.isValidating.set(true);
    this.addLog('info', 'Starting data consistency validation...');

    this.masterDataService.validateDataConsistency()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.consistencyReport.set(response.report);
          this.isValidating.set(false);
          
          if (response.report.is_consistent) {
            this.addLog('success', '✅ All data is consistent!');
          } else {
            this.addLog('warning', `⚠️ Found ${response.report.total_issues} consistency issues`);
          }
        },
        error: (error) => {
          this.isValidating.set(false);
          this.addLog('error', `Validation failed: ${error.message || error}`);
        }
      });
  }

  reloadMasterData() {
    this.isReloading.set(true);
    this.addLog('info', 'Reloading master data from CSV files...');

    this.masterDataService.reloadMasterData()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.isReloading.set(false);
          this.masterDataConfig.set(response.config);
          this.addLog('success', '🔄 Master data reloaded successfully');
          
          // Refresh summary after reload
          this.loadMasterDataSummary();
        },
        error: (error) => {
          this.isReloading.set(false);
          this.addLog('error', `Reload failed: ${error.message || error}`);
        }
      });
  }

  exportMasterData() {
    this.addLog('info', 'Exporting master data configuration...');

    this.masterDataService.exportMasterData()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          // Download the exported data as a JSON file
          const blob = new Blob([response.export_data], { type: 'application/json' });
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `master-data-config-${new Date().toISOString().split('T')[0]}.json`;
          link.click();
          window.URL.revokeObjectURL(url);
          
          this.addLog('success', '📤 Master data configuration exported');
        },
        error: (error) => {
          this.addLog('error', `Export failed: ${error.message || error}`);
        }
      });
  }

  getOverallStatus(): 'success' | 'warning' | 'error' {
    const report = this.consistencyReport();
    if (!report) return 'warning';
    return this.masterDataService.getIssueSeverity(report.total_issues);
  }

  getStatusIcon(): string {
    const status = this.getOverallStatus();
    switch (status) {
      case 'success': return '✅';
      case 'warning': return '⚠️';
      case 'error': return '❌';
      default: return '🔍';
    }
  }

  private addLog(level: 'info' | 'success' | 'warning' | 'error', message: string) {
    const logs = this.activityLog();
    logs.push({
      timestamp: new Date(),
      level,
      message
    });
    this.activityLog.set([...logs]);
  }
}