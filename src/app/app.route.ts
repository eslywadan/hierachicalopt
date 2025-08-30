import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';

export const routes: Routes = [
  {
    path: '',
    component: HomeComponent
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