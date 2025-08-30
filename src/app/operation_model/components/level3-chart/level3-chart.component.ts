// ===FILE: src/app/operation_model/components/level3-chart/level3-chart.component.ts===

import { Component, Input, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import { DataPoint } from '../../models/level3-lstm-model';

Chart.register(...registerables);

@Component({
  selector: 'app-level3-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="chart-container">
      <canvas #chartCanvas></canvas>
    </div>
  `,
  styles: [`
    .chart-container {
      position: relative;
      height: 400px;
      width: 100%;
      padding: 20px;
    }
  `]
})
export class Level3ChartComponent implements OnChanges, AfterViewInit {
  @Input() data: DataPoint[] = [];
  @Input() chartType: 'wip' | 'throughput' | 'cycleTime' | 'compliance' = 'wip';
  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;
  
  private chart: Chart | null = null;

  ngAfterViewInit(): void {
    this.createChart();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] || changes['chartType']) {
      this.updateChart();
    }
  }

  private createChart(): void {
    if (!this.chartCanvas) return;

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const config = this.getChartConfig();
    this.chart = new Chart(ctx, config);
  }

  private updateChart(): void {
    if (!this.chart) {
      this.createChart();
      return;
    }

    const config = this.getChartConfig();
    this.chart.data = config.data;
    if (config.options) {
      this.chart.options = config.options;
    }
    this.chart.update();
  }

  private getChartConfig(): ChartConfiguration {
    const labels = this.data.map(d => d.date.toLocaleDateString());
    let datasets: any[] = [];
    let yAxisLabel = '';

    switch (this.chartType) {
      case 'wip':
        datasets = [{
          label: 'WIP',
          data: this.data.map(d => d.wip),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1
        }, {
          label: 'Finished Goods',
          data: this.data.map(d => d.finishedGoods),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Units';
        break;

      case 'throughput':
        datasets = [{
          label: 'Throughput',
          data: this.data.map(d => d.throughput),
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Units/Day';
        break;

      case 'cycleTime':
        datasets = [{
          label: 'Cycle Time',
          data: this.data.map(d => d.cycleTime),
          borderColor: 'rgb(255, 206, 86)',
          backgroundColor: 'rgba(255, 206, 86, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Days';
        break;

      case 'compliance':
        datasets = [{
          label: "Little's Law Compliance",
          data: this.data.map(d => d.littlesLawCompliance || 0),
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.1
        }];
        yAxisLabel = 'Compliance %';
        break;
    }

    return {
      type: 'line',
      data: {
        labels,
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: `Level 3 ${this.chartType.charAt(0).toUpperCase() + this.chartType.slice(1)} Analysis`
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            title: {
              display: true,
              text: yAxisLabel
            }
          },
          x: {
            title: {
              display: true,
              text: 'Date'
            }
          }
        }
      }
    };
  }
}
