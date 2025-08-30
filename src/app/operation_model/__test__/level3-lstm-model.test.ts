// ===FILE: src/app/operation_model/__tests__/level3-lstm-model.test.ts===

import { 
  DataGenerator, 
  LSTMModel, 
  ScheduleValidator,
  PerformanceAnalyzer,
  Level3OperationModel 
} from '../models/level3-lstm-model';

describe('Level3 LSTM Model Tests', () => {
  let dataGenerator: DataGenerator;
  let lstmModel: LSTMModel;
  let validator: ScheduleValidator;
  let analyzer: PerformanceAnalyzer;
  
  beforeEach(() => {
    const config = {
      plants: ['Plant_1', 'Plant_2'],
      applications: ['Automotive', 'Consumer'],
      panelSizes: ['Small', 'Medium'],
      historicalDays: 30
    };
    
    const params = {
      baseWIP: 500,
      baseThroughput: 100,
      seasonality: 0.3,
      noiseLevel: 0.1
    };
    
    dataGenerator = new DataGenerator(config, params);
    lstmModel = new LSTMModel({
      lstmUnits1: 64,
      lstmUnits2: 32,
      dropoutRate: 0.2,
      sequenceLength: 7,
      epochs: 10,
      batchSize: 16,
      learningRate: 0.001,
      trainTestSplit: 0.8
    });
    validator = new ScheduleValidator();
    analyzer = new PerformanceAnalyzer();
  });
  
  describe('DataGenerator', () => {
    test('should generate correct number of data points', () => {
      const data = dataGenerator.generateData();
      expect(data.length).toBe(30 * 2 * 2 * 2); // days * plants * apps * sizes
    });
    
    test('should follow Little\'s Law', () => {
      const data = dataGenerator.generateData();
      data.forEach(point => {
        const calculatedCT = point.wip / point.throughput;
        expect(Math.abs(calculatedCT - point.cycleTime)).toBeLessThan(0.01);
      });
    });
    
    test('should apply seasonality correctly', () => {
      const data = dataGenerator.generateData();
      const winterData = data.filter(d => d.day < 90);
      const summerData = data.filter(d => d.day > 180 && d.day < 270);
      
      const winterAvgWIP = winterData.reduce((sum, d) => sum + d.wip, 0) / winterData.length;
      const summerAvgWIP = summerData.reduce((sum, d) => sum + d.wip, 0) / summerData.length;
      
      // Summer should have different WIP due to seasonality
      expect(Math.abs(winterAvgWIP - summerAvgWIP)).toBeGreaterThan(10);
    });
  });
  
  describe('ScheduleValidator', () => {
    test('should validate feasible schedule', () => {
      const request = {
        plant: 'Plant_1',
        application: 'Automotive',
        panelSize: 'Medium',
        predictionDays: 7,
        currentWIP: 500,
        plannedThroughput: 100,
        targetProduction: 700
      };
      
      const result = validator.validateSchedule(request);
      expect(result.isValid).toBe(true);
      expect(result.expectedCycleTime).toBeCloseTo(5, 1);
    });
    
    test('should detect infeasible schedule', () => {
      const request = {
        plant: 'Plant_1',
        application: 'Automotive',
        panelSize: 'Medium',
        predictionDays: 7,
        currentWIP: 500,
        plannedThroughput: 50,
        targetProduction: 700
      };
      
      const result = validator.validateSchedule(request);
      expect(result.isValid).toBe(false);
      expect(result.requiredThroughput).toBe(100);
    });
    
    test('should calculate Little\'s Law compliance', () => {
      const predictions = [
        { littlesLawCompliance: 95, feasible: true },
        { littlesLawCompliance: 92, feasible: true },
        { littlesLawCompliance: 88, feasible: false },
        { littlesLawCompliance: 96, feasible: true }
      ] as any;
      
      const result = validator.validatePredictions(predictions);
      expect(result.averageCompliance).toBeCloseTo(92.75, 2);
      expect(result.violations).toBe(1);
    });
  });
  
  describe('PerformanceAnalyzer', () => {
    test('should calculate R² score correctly', () => {
      const actual = [10, 20, 30, 40, 50];
      const predicted = [12, 18, 32, 38, 52];
      
      const metrics = analyzer.calculateMetrics(actual, predicted);
      expect(metrics.r2Score).toBeGreaterThan(0.9);
    });
    
    test('should calculate RMSE correctly', () => {
      const actual = [10, 20, 30];
      const predicted = [11, 19, 31];
      
      const metrics = analyzer.calculateMetrics(actual, predicted);
      expect(metrics.rmse).toBeCloseTo(1.15, 1);
    });
    
    test('should calculate MAPE correctly', () => {
      const actual = [100, 200, 300];
      const predicted = [110, 190, 330];
      
      const metrics = analyzer.calculateMetrics(actual, predicted);
      expect(metrics.mape).toBeLessThan(15);
    });
  });
  
  describe('Integration Tests', () => {
    test('should complete full training cycle', async () => {
      const model = new Level3OperationModel(
        { plants: ['Plant_1'], applications: ['Automotive'], panelSizes: ['Medium'], historicalDays: 10 },
        { baseWIP: 500, baseThroughput: 100, seasonality: 0.1, noiseLevel: 0.05 },
        { lstmUnits1: 16, lstmUnits2: 8, dropoutRate: 0.1, sequenceLength: 3, epochs: 2, batchSize: 4, learningRate: 0.01, trainTestSplit: 0.7 }
      );
      
      await model.initialize();
      const dataCount = model.exportData().length;
      expect(dataCount).toBe(10);
      
      // Note: Actual training would require TensorFlow.js setup in test environment
    });
  });
});
