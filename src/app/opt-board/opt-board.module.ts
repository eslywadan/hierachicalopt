import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Routes } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { OptBoardComponent } from './opt-board.component';
import { TFTLCDDashboardComponent } from './components/tft-lcd-dashboard.component';

const routes: Routes = [
  {
    path: '',
    component: OptBoardComponent
  },
  {
    path: 'tft-lcd',
    component: TFTLCDDashboardComponent
  }
];

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    RouterModule.forChild(routes),
    OptBoardComponent,
    TFTLCDDashboardComponent
  ]
})
export class OptBoardModule { }