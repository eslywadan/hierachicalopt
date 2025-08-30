import { Injectable, NgZone } from '@angular/core';

// Reference Chart from the global window object where it's registered in main.ts
declare const Chart: any;

@Injectable({
  providedIn: 'root'
})
export class ChartService {
  private charts: Map<string, any> = new Map();

  constructor(private ngZone: NgZone) {}

  createLineChart(canvasId: string, data: any[], xKey: string, yKey: string, label: string): void {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      console.warn(`Canvas element with id ${canvasId} not found`);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.ngZone.runOutsideAngular(() => {
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map(item => item[xKey]),
          datasets: [{
            label: label,
            data: data.map(item => item[yKey]),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            tension: 0.1,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          aspectRatio: 1.5,
          plugins: {
            legend: {
              position: 'top' as const,
            },
            tooltip: {
              mode: 'index',
              intersect: false,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: xKey
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: label
              }
            }
          },
          interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
          }
        }
      });

      this.charts.set(canvasId, chart);
    });
  }

  createBarChart(canvasId: string, data: any[], xKey: string, yKey: string, label: string): void {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      console.warn(`Canvas element with id ${canvasId} not found`);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.ngZone.runOutsideAngular(() => {
      const chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: data.map(item => item[xKey]),
          datasets: [{
            label: label,
            data: data.map(item => item[yKey]),
            backgroundColor: 'rgba(54, 162, 235, 0.8)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          aspectRatio: 1.5,
          plugins: {
            legend: {
              position: 'top' as const,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: xKey
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: label
              },
              beginAtZero: true
            }
          }
        }
      });

      this.charts.set(canvasId, chart);
    });
  }

  createPieChart(canvasId: string, data: any[], labelKey: string, valueKey: string): void {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      console.warn(`Canvas element with id ${canvasId} not found`);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const colors = [
      'rgba(255, 99, 132, 0.8)',
      'rgba(54, 162, 235, 0.8)',
      'rgba(255, 205, 86, 0.8)',
      'rgba(75, 192, 192, 0.8)',
      'rgba(153, 102, 255, 0.8)',
      'rgba(255, 159, 64, 0.8)'
    ];

    this.ngZone.runOutsideAngular(() => {
      const chart = new Chart(ctx, {
        type: 'pie',
        data: {
          labels: data.map(item => item[labelKey]),
          datasets: [{
            data: data.map(item => item[valueKey]),
            backgroundColor: colors.slice(0, data.length),
            borderColor: colors.slice(0, data.length).map(color => color.replace('0.8', '1')),
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          aspectRatio: 1.5,
          plugins: {
            legend: {
              position: 'bottom' as const,
            },
            tooltip: {
              callbacks: {
                label: (context: any) => {
                  const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
                  const percentage = ((context.raw / total) * 100).toFixed(1);
                  return `${context.label}: ${context.formattedValue} (${percentage}%)`;
                }
              }
            }
          }
        }
      });

      this.charts.set(canvasId, chart);
    });
  }

  createMultiLineChart(canvasId: string, data: any[], xKey: string, datasets: Array<{key: string, label: string, color: string}>): void {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      console.warn(`Canvas element with id ${canvasId} not found`);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.ngZone.runOutsideAngular(() => {
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map(item => item[xKey]),
          datasets: datasets.map(dataset => ({
            label: dataset.label,
            data: data.map(item => item[dataset.key]),
            borderColor: dataset.color,
            backgroundColor: dataset.color.replace('1)', '0.1)'),
            tension: 0.1,
            fill: false
          }))
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          aspectRatio: 1.5,
          plugins: {
            legend: {
              position: 'top' as const,
            },
            tooltip: {
              mode: 'index',
              intersect: false,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: xKey
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Value'
              }
            }
          },
          interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
          }
        }
      });

      this.charts.set(canvasId, chart);
    });
  }

  createAreaChart(canvasId: string, data: any[], xKey: string, datasets: Array<{key: string, label: string, color: string}>): void {
    this.destroyChart(canvasId);
    
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      console.warn(`Canvas element with id ${canvasId} not found`);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.ngZone.runOutsideAngular(() => {
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map(item => item[xKey]),
          datasets: datasets.map(dataset => ({
            label: dataset.label,
            data: data.map(item => item[dataset.key]),
            borderColor: dataset.color,
            backgroundColor: dataset.color.replace('1)', '0.6)'),
            tension: 0.1,
            fill: true
          }))
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          aspectRatio: 1.5,
          plugins: {
            legend: {
              position: 'top' as const,
            },
            tooltip: {
              mode: 'index',
              intersect: false,
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: xKey
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Value'
              },
              stacked: true
            }
          },
          interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
          }
        }
      });

      this.charts.set(canvasId, chart);
    });
  }

  updateChartData(canvasId: string, _newData: any[]): void {
    const chart = this.charts.get(canvasId);
    if (chart) {
      this.ngZone.runOutsideAngular(() => {
        chart.update();
      });
    }
  }

  destroyChart(canvasId: string): void {
    const chart = this.charts.get(canvasId);
    if (chart) {
      this.ngZone.runOutsideAngular(() => {
        chart.destroy();
      });
      this.charts.delete(canvasId);
    }
  }

  destroyAllCharts(): void {
    this.charts.forEach((_chart, id) => {
      this.destroyChart(id);
    });
  }

  // Helper method to format currency values
  formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  }

  // Helper method to format percentage values
  formatPercentage(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }
}