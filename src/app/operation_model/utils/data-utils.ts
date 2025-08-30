// ===FILE: src/app/operation_model/utils/data-utils.ts===

/**
 * Utility functions for data processing and manipulation
 */

export class DataUtils {
  /**
   * Normalize time series data
   */
  static normalize(data: number[], method: 'minmax' | 'zscore' = 'minmax'): number[] {
    if (method === 'minmax') {
      const min = Math.min(...data);
      const max = Math.max(...data);
      const range = max - min;
      return data.map(v => (v - min) / range);
    } else {
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const std = Math.sqrt(
        data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.length
      );
      return data.map(v => (v - mean) / std);
    }
  }
  
  /**
   * Denormalize data
   */
  static denormalize(
    normalizedData: number[],
    originalMin: number,
    originalMax: number
  ): number[] {
    const range = originalMax - originalMin;
    return normalizedData.map(v => v * range + originalMin);
  }
  
  /**
   * Create sliding windows for time series
   */
  static createSlidingWindows<T>(
    data: T[],
    windowSize: number,
    stride: number = 1
  ): T[][] {
    const windows: T[][] = [];
    
    for (let i = 0; i <= data.length - windowSize; i += stride) {
      windows.push(data.slice(i, i + windowSize));
    }
    
    return windows;
  }
  
  /**
   * Calculate moving average
   */
  static movingAverage(data: number[], windowSize: number): number[] {
    const result: number[] = [];
    
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = data.slice(start, i + 1);
      const avg = window.reduce((a, b) => a + b, 0) / window.length;
      result.push(avg);
    }
    
    return result;
  }
  
  /**
   * Detect anomalies using z-score
   */
  static detectAnomalies(
    data: number[],
    threshold: number = 3
  ): { indices: number[]; values: number[] } {
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const std = Math.sqrt(
      data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.length
    );
    
    const anomalies = {
      indices: [] as number[],
      values: [] as number[]
    };
    
    data.forEach((value, index) => {
      const zScore = Math.abs((value - mean) / std);
      if (zScore > threshold) {
        anomalies.indices.push(index);
        anomalies.values.push(value);
      }
    });
    
    return anomalies;
  }
  
  /**
   * Interpolate missing values
   */
  static interpolate(data: (number | null)[]): number[] {
    const result = [...data];
    
    for (let i = 0; i < result.length; i++) {
      if (result[i] === null) {
        // Find previous and next non-null values
        let prevIndex = i - 1;
        let nextIndex = i + 1;
        
        while (prevIndex >= 0 && result[prevIndex] === null) prevIndex--;
        while (nextIndex < result.length && result[nextIndex] === null) nextIndex++;
        
        if (prevIndex >= 0 && nextIndex < result.length) {
          // Linear interpolation
          const prevValue = result[prevIndex] as number;
          const nextValue = result[nextIndex] as number;
          const steps = nextIndex - prevIndex;
          const step = (nextValue - prevValue) / steps;
          
          result[i] = prevValue + step * (i - prevIndex);
        } else if (prevIndex >= 0) {
          result[i] = result[prevIndex];
        } else if (nextIndex < result.length) {
          result[i] = result[nextIndex];
        } else {
          result[i] = 0; // Default value
        }
      }
    }
    
    return result as number[];
  }
  
  /**
   * Calculate correlation between two series
   */
  static correlation(x: number[], y: number[]): number {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const correlation = (n * sumXY - sumX * sumY) /
      Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return correlation;
  }
}
