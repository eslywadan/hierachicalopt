// ===FILE: src/app/operation_model/database/models.ts===

import { DataTypes, Model, Sequelize } from 'sequelize';

/**
 * Database models for Level 3 Operation Model
 */

export class TrainingData extends Model {
  public id!: number;
  public day!: number;
  public date!: Date;
  public plant!: string;
  public application!: string;
  public panelSize!: string;
  public wip!: number;
  public throughput!: number;
  public cycleTime!: number;
  public finishedGoods!: number;
  public semiFinishedGoods!: number;
  
  static initialize(sequelize: Sequelize) {
    TrainingData.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true
      },
      day: DataTypes.INTEGER,
      date: DataTypes.DATE,
      plant: DataTypes.STRING,
      application: DataTypes.STRING,
      panelSize: DataTypes.STRING,
      wip: DataTypes.FLOAT,
      throughput: DataTypes.FLOAT,
      cycleTime: DataTypes.FLOAT,
      finishedGoods: DataTypes.FLOAT,
      semiFinishedGoods: DataTypes.FLOAT
    }, {
      sequelize,
      tableName: 'training_data'
    });
  }
}

export class PredictionLog extends Model {
  public id!: number;
  public requestId!: string;
  public timestamp!: Date;
  public plant!: string;
  public application!: string;
  public panelSize!: string;
  public predictionDays!: number;
  public predictions!: any;
  public validation!: any;
  public compliance!: number;
  
  static initialize(sequelize: Sequelize) {
    PredictionLog.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true
      },
      requestId: DataTypes.STRING,
      timestamp: DataTypes.DATE,
      plant: DataTypes.STRING,
      application: DataTypes.STRING,
      panelSize: DataTypes.STRING,
      predictionDays: DataTypes.INTEGER,
      predictions: DataTypes.JSON,
      validation: DataTypes.JSON,
      compliance: DataTypes.FLOAT
    }, {
      sequelize,
      tableName: 'prediction_logs'
    });
  }
}

export class ModelMetrics extends Model {
  public id!: number;
  public modelVersion!: string;
  public timestamp!: Date;
  public r2Score!: number;
  public rmse!: number;
  public mae!: number;
  public mape!: number;
  public littlesLawCompliance!: number;
  
  static initialize(sequelize: Sequelize) {
    ModelMetrics.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true
      },
      modelVersion: DataTypes.STRING,
      timestamp: DataTypes.DATE,
      r2Score: DataTypes.FLOAT,
      rmse: DataTypes.FLOAT,
      mae: DataTypes.FLOAT,
      mape: DataTypes.FLOAT,
      littlesLawCompliance: DataTypes.FLOAT
    }, {
      sequelize,
      tableName: 'model_metrics'
    });
  }
}

// Database connection setup
export class Database {
  private sequelize: Sequelize;
  
  constructor(config?: any) {
    this.sequelize = new Sequelize(config || {
      dialect: 'sqlite',
      storage: './data/level3.db',
      logging: false
    });
    
    this.initializeModels();
  }
  
  private initializeModels(): void {
    TrainingData.initialize(this.sequelize);
    PredictionLog.initialize(this.sequelize);
    ModelMetrics.initialize(this.sequelize);
  }
  
  async sync(): Promise<void> {
    await this.sequelize.sync();
  }
  
  getSequelize(): Sequelize {
    return this.sequelize;
  }
}