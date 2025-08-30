import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="home-container">
      <h2>Welcome to Hierarchical Optimization</h2>
      <p>Choose a module to get started:</p>
      
      <div class="modules-grid">
        <div class="module-card">
          <h3>Opt Board</h3>
          <p>General optimization dashboard with TFT LCD analysis tools.</p>
          <a routerLink="/opt-board" class="module-button">Open Opt Board →</a>
        </div>
        
        <div class="module-card">
          <h3>Operation Model</h3>
          <p>Level 3 LSTM neural network for production planning and schedule validation.</p>
          <a routerLink="/operation-model" class="module-button">Open Operation Model →</a>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .home-container {
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
      text-align: center;
    }

    .modules-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 2rem;
      margin-top: 2rem;
    }

    .module-card {
      background: white;
      border-radius: 8px;
      padding: 2rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      text-align: left;
      transition: transform 0.2s;
    }

    .module-card:hover {
      transform: translateY(-2px);
    }

    .module-button {
      display: inline-block;
      background-color: #1976d2;
      color: white;
      padding: 0.75rem 1.5rem;
      text-decoration: none;
      border-radius: 4px;
      margin-top: 1rem;
      transition: background-color 0.2s;
    }

    .module-button:hover {
      background-color: #1565c0;
    }

    h2 {
      color: #333;
      margin-bottom: 1rem;
    }

    h3 {
      color: #1976d2;
      margin-bottom: 0.5rem;
    }

    p {
      color: #666;
      line-height: 1.6;
    }
  `]
})
export class HomeComponent { }