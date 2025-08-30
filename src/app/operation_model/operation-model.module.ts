// ===FILE: src/app/operation_model/operation-model.module.ts===

import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Routes } from '@angular/router';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';

// Removed Material imports - will add back when compatible version available

// Components
import { Level3DashboardComponent } from './components/level3-dashboard/level3-dashboard.component';
import { Level3ChartComponent } from './components/level3-chart/level3-chart.component';
import { BackendDashboardComponent } from './components/backend-dashboard/backend-dashboard.component';
import { DataManagementComponent } from './components/data-management/data-management.component';

// Services
import { Level3OperationService } from './services/level3-operation.service';
import { DATA_CONFIG, DATA_GENERATOR_PARAMS, LSTM_CONFIG, DataConfig, DataGeneratorParams, LSTMConfig } from './models/level3-lstm-model';

const routes: Routes = [
  {
    path: '',
    redirectTo: 'backend',
    pathMatch: 'full'
  },
  {
    path: 'backend',
    component: BackendDashboardComponent
  },
  {
    path: 'level3',
    component: Level3DashboardComponent
  },
  {
    path: 'data-management',
    component: DataManagementComponent
  }
];

@NgModule({
  declarations: [],
  imports: [
    CommonModule,
    ReactiveFormsModule,
    FormsModule,
    RouterModule.forChild(routes),
    // Standalone components
    Level3DashboardComponent,
    Level3ChartComponent,
    BackendDashboardComponent,
    DataManagementComponent
  ],
  providers: [
    Level3OperationService,
    {
      provide: DATA_CONFIG,
      useValue: {
        plants: ['Plant_1', 'Plant_2', 'Plant_3'],
        applications: ['Automotive', 'Consumer_Electronics', 'Industrial'],
        panelSizes: ['Small', 'Medium', 'Large'],
        historicalDays: 545  // User specified value for optimal performance
      } as DataConfig
    },
    {
      provide: DATA_GENERATOR_PARAMS,
      useValue: {
        baseWIP: 1000,
        baseThroughput: 500,
        seasonality: 0.2,
        noiseLevel: 0.1
      } as DataGeneratorParams
    },
    {
      provide: LSTM_CONFIG,
      useValue: {
        lstmUnits1: 16,  // Very small for fast training
        lstmUnits2: 8,   // Minimal second layer
        dropoutRate: 0.1,
        sequenceLength: 5,   // Very short sequences
        epochs: 3,      // Minimal epochs
        batchSize: 4,   // Very small batch size
        learningRate: 0.01,  // Much higher learning rate
        trainTestSplit: 0.8
      } as LSTMConfig
    }
  ]
})
export class OperationModelModule { }
// ===FILE: src/app/operation_model/utils/littles-law-calculator.ts===