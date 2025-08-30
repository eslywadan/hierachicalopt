import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-opt-board',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="opt-board-container">
      <h2>Optimization Board</h2>
      <p>Comprehensive manufacturing analytics and optimization dashboard.</p>
      <div class="features-grid">
        <div class="feature-card">
          <div class="feature-icon">📊</div>
          <h3>TFT-LCD Analytics</h3>
          <p>Real-time manufacturing analytics with interactive charts and KPI monitoring.</p>
          <a href="/opt-board/tft-lcd" class="feature-button">Open TFT-LCD Dashboard</a>
        </div>

        <div class="feature-card">
          <div class="feature-icon">🧠</div>
          <h3>LSTM Operation Model</h3>
          <p>Advanced machine learning model for production planning and schedule validation using Little's Law.</p>
          <a href="/operation-model" class="feature-button">Open LSTM Model</a>
        </div>

        <div class="feature-card">
          <div class="feature-icon">⚙️</div>
          <h3>Optimization Analysis</h3>
          <p>Advanced optimization tools and analysis capabilities for manufacturing processes.</p>
          <a href="#" class="feature-button disabled">Coming Soon</a>
        </div>

        <div class="feature-card">
          <div class="feature-icon">🔄</div>
          <h3>Data Processing</h3>
          <p>Automated data processing utilities with CSV import and real-time analysis.</p>
          <a href="#" class="feature-button disabled">Coming Soon</a>
        </div>
      </div>

      <div class="integration-info">
        <h3>🔗 Integrated Platform Features</h3>
        <div class="integration-grid">
          <div class="integration-item">
            <strong>Unified Framework:</strong> Both dashboards built with Angular & TypeScript
          </div>
          <div class="integration-item">
            <strong>Shared Data:</strong> Common data processing and visualization services
          </div>
          <div class="integration-item">
            <strong>Chart.js Integration:</strong> Consistent charting across all components
          </div>
          <div class="integration-item">
            <strong>Responsive Design:</strong> Optimized for desktop and mobile viewing
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .opt-board-container {
      padding: 2rem;
      max-width: 1400px;
      margin: 0 auto;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .features-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 2rem;
      margin: 2rem 0;
    }

    .feature-card {
      background: white;
      border-radius: 12px;
      padding: 2rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      text-align: center;
      transition: transform 0.2s, box-shadow 0.2s;
    }

    .feature-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .feature-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
    }

    .feature-card h3 {
      color: #2d3748;
      margin: 1rem 0;
      font-size: 1.25rem;
      font-weight: 600;
    }

    .feature-card p {
      color: #718096;
      margin-bottom: 1.5rem;
      line-height: 1.5;
    }

    .feature-button {
      display: inline-block;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 0.75rem 1.5rem;
      text-decoration: none;
      border-radius: 6px;
      font-weight: 500;
      transition: all 0.2s;
    }

    .feature-button:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .feature-button.disabled {
      background: #e2e8f0;
      color: #a0aec0;
      cursor: not-allowed;
    }

    .feature-button.disabled:hover {
      transform: none;
      box-shadow: none;
    }

    .integration-info {
      background: #f7fafc;
      border-radius: 12px;
      padding: 2rem;
      margin-top: 3rem;
      border-left: 4px solid #667eea;
    }

    .integration-info h3 {
      color: #2d3748;
      margin: 0 0 1.5rem 0;
      font-size: 1.25rem;
      font-weight: 600;
    }

    .integration-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
    }

    .integration-item {
      background: white;
      padding: 1rem;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      color: #4a5568;
      line-height: 1.5;
    }

    .integration-item strong {
      color: #2d3748;
      display: block;
      margin-bottom: 0.5rem;
    }

    h2 {
      color: #2d3748;
      font-size: 2.25rem;
      font-weight: 700;
      margin: 0 0 0.5rem 0;
    }

    h2 + p {
      color: #718096;
      font-size: 1.1rem;
      margin: 0 0 2rem 0;
    }
  `]
})
export class OptBoardComponent { }