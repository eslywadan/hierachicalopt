import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, firstValueFrom } from 'rxjs';

export interface TFTLCDRecord {
  date: Date;
  plant: string;
  panel_size: string;
  market_segment: string;
  production_volume: number;
  capacity_utilization: number;
  yield_rate: number;
  defect_rate: number;
  unit_price: number;
  revenue: number;
  material_cost: number;
  supply_disruption: boolean;
  delivery_time: number;
}

export interface DataSummary {
  mainRecords: number;
  dateRange: {
    start: Date;
    end: Date;
    totalWeeks: number;
  };
  plants: Array<{value: string; label: string}>;
  panelSizes: Array<{value: string; label: string}>;
  marketSegments: Array<{value: string; label: string}>;
}

export interface KPIData {
  totalRevenue: number;
  avgYield: number;
  avgCapacity: number;
  disruptions: number;
}

export interface TFTLCDData {
  dataSummary: DataSummary;
  weeklyTrends: any[];
  plantPerformance: any[];
  sizeAnalysis: any[];
  supplyChainData: any[];
  qualityMetrics: any[];
  kpis: KPIData;
}

@Injectable({
  providedIn: 'root'
})
export class TFTLCDDataService {
  private dataCache$ = new BehaviorSubject<TFTLCDData | null>(null);
  private lastLoadTime = 0;
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  constructor(private http: HttpClient) {}

  async loadTFTLCDData(): Promise<TFTLCDData> {
    // Return cached data if recent
    const cached = this.dataCache$.value;
    if (cached && (Date.now() - this.lastLoadTime) < this.cacheTimeout) {
      return cached;
    }

    try {
      console.log('🔄 Loading TFT-LCD data...');
      
      // Load CSV data with fallback to synthetic data
      const [mainData, componentData, marketData] = await Promise.allSettled([
        this.loadCSV('/assets/data/tft_lcd_main_data.csv'),
        this.loadCSV('/assets/data/tft_lcd_component_data.csv'),
        this.loadCSV('/assets/data/tft_lcd_market_data.csv')
      ]).then(results => [
        results[0].status === 'fulfilled' ? results[0].value : [],
        results[1].status === 'fulfilled' ? results[1].value : [],
        results[2].status === 'fulfilled' ? results[2].value : []
      ]);

      console.log('📊 Data loaded:', {
        main: mainData.length,
        component: componentData.length,
        market: marketData.length
      });

      // Process data
      const processedData = this.processRawData(mainData, componentData, marketData);
      
      this.dataCache$.next(processedData);
      this.lastLoadTime = Date.now();
      
      return processedData;
    } catch (error) {
      console.error('Error loading TFT-LCD data:', error);
      throw error;
    }
  }

  private async loadCSV(url: string): Promise<any[]> {
    try {
      console.log(`Attempting to load CSV from: ${url}`);
      const csvText = await firstValueFrom(this.http.get(url, { responseType: 'text' }));
      const parsed = this.parseCSV(csvText);
      console.log(`Successfully loaded ${parsed.length} records from ${url}`);
      return parsed;
    } catch (error) {
      console.warn(`Failed to load ${url}:`, error);
      console.log(`Generating synthetic data for ${url}`);
      return this.generateSyntheticData(url);
    }
  }

  private parseCSV(csvText: string): any[] {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim());
      const row: any = {};
      
      headers.forEach((header, index) => {
        const value = values[index] || '';
        
        // Convert to appropriate data types
        if (header.toLowerCase().includes('date') || header === 'Date') {
          // Handle date parsing more robustly
          const dateValue = new Date(value);
          row[header] = isNaN(dateValue.getTime()) ? null : dateValue;
        } else if (!isNaN(Number(value)) && value !== '') {
          row[header] = Number(value);
        } else if (value.toLowerCase() === 'true' || value.toLowerCase() === 'false') {
          row[header] = value.toLowerCase() === 'true';
        } else {
          row[header] = value;
        }
      });
      
      data.push(row);
    }

    return data;
  }

  private generateSyntheticData(url: string): TFTLCDRecord[] {
    console.log(`Generating synthetic data for ${url}`);
    
    const baseData: TFTLCDRecord[] = [];
    const plants = ['Plant_A', 'Plant_B', 'Plant_C'];
    const panelSizes = ['Small', 'Medium', 'Large', 'Extra_Large'];
    const marketSegments = ['Automotive', 'Consumer_Electronics', 'Industrial'];
    
    const startDate = new Date('2024-01-01');
    const endDate = new Date('2024-12-31');
    const daysBetween = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

    for (let day = 0; day < daysBetween; day += 7) { // Weekly data
      const currentDate = new Date(startDate.getTime() + day * 24 * 60 * 60 * 1000);
      
      plants.forEach(plant => {
        panelSizes.forEach(size => {
          marketSegments.forEach(segment => {
            const baseVolume = this.getBaseVolume(size);
            const seasonality = 1 + 0.2 * Math.sin(2 * Math.PI * day / 365);
            const productionVolume = Math.floor(baseVolume * seasonality * (0.8 + Math.random() * 0.4));
            const capacityUtilization = 0.7 + Math.random() * 0.25;
            const yieldRate = 0.85 + Math.random() * 0.1;
            const defectRate = Math.random() * 0.05;
            const unitPrice = this.getUnitPrice(size, segment);
            const materialCost = this.getMaterialCost(size);
            const supplyDisruption = Math.random() < 0.1;
            const deliveryTime = 5 + Math.random() * 10;
            
            baseData.push({
              date: currentDate,
              plant: plant,
              panel_size: size,
              market_segment: segment,
              production_volume: productionVolume,
              capacity_utilization: capacityUtilization,
              yield_rate: yieldRate,
              defect_rate: defectRate,
              unit_price: unitPrice,
              revenue: productionVolume * unitPrice * yieldRate, // Calculate revenue directly
              material_cost: materialCost,
              supply_disruption: supplyDisruption,
              delivery_time: deliveryTime
            });
          });
        });
      });
    }

    return baseData;
  }

  private getBaseVolume(size: string): number {
    const volumes = {
      'Small': 1000,
      'Medium': 800,
      'Large': 600,
      'Extra_Large': 400
    };
    return volumes[size as keyof typeof volumes] || 500;
  }

  private getUnitPrice(size: string, segment: string): number {
    const basePrices = {
      'Small': 50,
      'Medium': 120,
      'Large': 250,
      'Extra_Large': 400
    };
    
    const segmentMultiplier = {
      'Automotive': 1.3,
      'Consumer_Electronics': 1.0,
      'Industrial': 1.5
    };
    
    const basePrice = basePrices[size as keyof typeof basePrices] || 100;
    const multiplier = segmentMultiplier[segment as keyof typeof segmentMultiplier] || 1.0;
    
    return basePrice * multiplier * (0.9 + Math.random() * 0.2);
  }

  private getMaterialCost(size: string): number {
    const baseCosts = {
      'Small': 25,
      'Medium': 60,
      'Large': 125,
      'Extra_Large': 200
    };
    return baseCosts[size as keyof typeof baseCosts] || 50;
  }

  private processRawData(mainData: any[], _componentData: any[], _marketData: any[]): TFTLCDData {
    console.log(`🔍 Processing data: ${mainData.length} main records available`);
    
    // Use mainData as primary source, supplement with component/market data if needed
    let data = mainData.length > 0 ? mainData : this.generateSyntheticData('fallback');
    
    console.log(`✅ Using ${data.length} records for processing`);
    
    // Calculate data summary
    const dataSummary = this.calculateDataSummary(data);
    
    // Generate aggregated views
    const weeklyTrends = this.generateWeeklyTrends(data);
    const plantPerformance = this.generatePlantPerformance(data);
    const sizeAnalysis = this.generateSizeAnalysis(data);
    const supplyChainData = this.generateSupplyChainData(data);
    const qualityMetrics = this.generateQualityMetrics(data);
    const kpis = this.calculateKPIs(data);

    return {
      dataSummary,
      weeklyTrends,
      plantPerformance,
      sizeAnalysis,
      supplyChainData,
      qualityMetrics,
      kpis
    };
  }

  private calculateDataSummary(data: any[]): DataSummary {
    // Extract dates, handling both 'Date' and 'date' columns, and filter out null values
    const dateColumn = data[0]?.Date ? 'Date' : 'date';
    const dates = data
      .map(d => d[dateColumn])
      .filter(d => d && !isNaN(new Date(d).getTime()))
      .map(d => new Date(d))
      .sort((a, b) => a.getTime() - b.getTime());
    
    // Extract plant information (try different column names)
    const plantColumn = data[0]?.Plant ? 'Plant' : (data[0]?.plant ? 'plant' : null);
    const plants = plantColumn ? [...new Set(data.map(d => d[plantColumn]).filter(p => p))] : ['Plant_A', 'Plant_B', 'Plant_C'];
    
    // Extract panel size information
    const panelSizeColumn = data[0]?.Panel_Size ? 'Panel_Size' : (data[0]?.panel_size ? 'panel_size' : null);
    const panelSizes = panelSizeColumn ? [...new Set(data.map(d => d[panelSizeColumn]).filter(s => s))] : ['Small', 'Medium', 'Large'];
    
    // Extract market segment information  
    const marketSegmentColumn = data[0]?.Market_Segment ? 'Market_Segment' : (data[0]?.market_segment ? 'market_segment' : null);
    const marketSegments = marketSegmentColumn ? [...new Set(data.map(d => d[marketSegmentColumn]).filter(m => m))] : ['Automotive', 'Consumer_Electronics', 'Industrial'];

    let startDate = new Date('2024-01-01');
    let endDate = new Date('2024-12-31');
    let totalWeeks = 52;
    
    if (dates.length > 0) {
      startDate = dates[0];
      endDate = dates[dates.length - 1];
      totalWeeks = Math.max(1, Math.floor((endDate.getTime() - startDate.getTime()) / (7 * 24 * 60 * 60 * 1000)));
    }

    console.log('📊 Data summary calculated:', {
      records: data.length,
      dates: dates.length,
      plants: plants.length,
      panelSizes: panelSizes.length,
      marketSegments: marketSegments.length,
      dateRange: { start: startDate, end: endDate, totalWeeks }
    });

    return {
      mainRecords: data.length,
      dateRange: {
        start: startDate,
        end: endDate,
        totalWeeks
      },
      plants: plants.map(p => ({ value: p, label: p })),
      panelSizes: panelSizes.map(s => ({ value: s, label: s })),
      marketSegments: marketSegments.map(m => ({ value: m, label: m }))
    };
  }

  private generateWeeklyTrends(data: any[]): any[] {
    const weeklyData = new Map();
    
    data.forEach(record => {
      const dateValue = record.Date || record.date;
      if (!dateValue) return;
      
      const date = new Date(dateValue);
      if (isNaN(date.getTime())) return;
      
      const week = this.getWeekKey(date);
      if (!weeklyData.has(week)) {
        weeklyData.set(week, {
          week,
          revenue: 0,
          production: 0,
          demand: 0,
          avg_yield: 0,
          avg_capacity: 0,
          count: 0
        });
      }
      
      const weekData = weeklyData.get(week);
      // Map CSV columns to our expected structure
      weekData.revenue += record.Revenue || record.revenue || 0;
      weekData.production += record.Actual_Production || record.production_volume || record.production || 0;
      weekData.demand += record.Forecasted_Demand || record.demand || ((record.Actual_Production || 0) * 1.1);
      weekData.avg_yield += record.Production_Yield || record.yield_rate || record.yield || 0;
      weekData.avg_capacity += record.Capacity_Utilization || record.capacity_utilization || record.capacity || 0;
      weekData.count++;
    });

    return Array.from(weeklyData.values()).map(week => ({
      ...week,
      avg_yield: week.avg_yield / week.count,
      avg_capacity: week.avg_capacity / week.count
    })).sort((a, b) => a.week.localeCompare(b.week));
  }

  private generatePlantPerformance(data: any[]): any[] {
    const plantData = new Map();
    
    data.forEach(record => {
      const plant = record.Plant || record.plant;
      if (!plant) return;
      
      if (!plantData.has(plant)) {
        plantData.set(plant, {
          plant,
          revenue: 0,
          yield: 0,
          capacity: 0,
          defects: 0,
          count: 0
        });
      }
      
      const pData = plantData.get(plant);
      pData.revenue += record.Revenue || record.revenue || 0;
      pData.yield += record.Production_Yield || record.yield_rate || record.yield || 0;
      pData.capacity += record.Capacity_Utilization || record.capacity_utilization || record.capacity || 0;
      pData.defects += record.Defect_Rate || record.defect_rate || record.defects || 0;
      pData.count++;
    });

    return Array.from(plantData.values()).map(plant => ({
      plant: plant.plant,
      revenue: plant.revenue,
      yield: plant.yield / plant.count,
      capacity: plant.capacity / plant.count,
      defects: plant.defects / plant.count
    }));
  }

  private generateSizeAnalysis(data: any[]): any[] {
    const sizeData = new Map();
    
    data.forEach(record => {
      const size = record.Panel_Size || record.panel_size;
      if (!size) return;
      
      if (!sizeData.has(size)) {
        sizeData.set(size, {
          size,
          revenue: 0,
          volume: 0,
          avg_price: 0,
          count: 0
        });
      }
      
      const sData = sizeData.get(size);
      sData.revenue += record.Revenue || record.revenue || 0;
      sData.volume += record.Actual_Production || record.production_volume || record.volume || 0;
      sData.avg_price += record.Unit_Selling_Price || record.unit_price || record.price || 0;
      sData.count++;
    });

    return Array.from(sizeData.values()).map(size => ({
      size: size.size,
      revenue: size.revenue,
      volume: size.volume,
      avg_price: size.avg_price / size.count
    }));
  }

  private generateSupplyChainData(data: any[]): any[] {
    const plantData = new Map();
    
    data.forEach(record => {
      const plant = record.Plant || record.plant;
      if (!plant) return;
      
      if (!plantData.has(plant)) {
        plantData.set(plant, {
          plant,
          disruptions: 0,
          total_deliveries: 0,
          on_time_deliveries: 0,
          avg_delivery: 0,
          count: 0
        });
      }
      
      const pData = plantData.get(plant);
      // Map CSV columns to our expected structure
      const hasDisruption = record.Supply_Disruptions || record.supply_disruption || record.disruptions || 0;
      if (hasDisruption > 0) pData.disruptions++;
      
      pData.total_deliveries++;
      const onTimeDelivery = record.On_Time_Delivery || record.on_time_delivery || 0;
      const leadTime = record.Lead_Time_Days || record.delivery_time || 0;
      
      // If on-time delivery rate is provided, use it; otherwise assume based on lead time
      if (onTimeDelivery > 0.7 || (leadTime > 0 && leadTime <= 7)) {
        pData.on_time_deliveries++;
      }
      
      pData.avg_delivery += leadTime;
      pData.count++;
    });

    return Array.from(plantData.values()).map(plant => ({
      plant: plant.plant,
      disruption_rate: (plant.disruptions / plant.count) * 100,
      avg_delivery: (plant.on_time_deliveries / plant.total_deliveries) * 100
    }));
  }

  private generateQualityMetrics(data: any[]): any[] {
    // Return plant-level quality metrics
    return this.generatePlantPerformance(data);
  }

  private calculateKPIs(data: any[]): KPIData {
    if (data.length === 0) {
      return { totalRevenue: 0, avgYield: 0, avgCapacity: 0, disruptions: 0 };
    }

    // Handle both CSV column names and synthetic data property names
    const totalRevenue = data.reduce((sum, record) => {
      const revenue = record.Revenue || record.revenue || 0;
      return sum + revenue;
    }, 0);
    
    const avgYield = data.reduce((sum, record) => {
      const yield_rate = record.Production_Yield || record.yield_rate || record.yield || 0;
      return sum + yield_rate;
    }, 0) / data.length;
    
    const avgCapacity = data.reduce((sum, record) => {
      const capacity = record.Capacity_Utilization || record.capacity_utilization || record.capacity || 0;
      return sum + capacity;
    }, 0) / data.length;
    
    const disruptions = data.filter(record => {
      const disruption = record.Supply_Disruptions || record.supply_disruption || record.disruptions;
      return disruption === true || disruption > 0;
    }).length;

    console.log('🔢 KPI Calculation Results:', {
      totalRevenue,
      avgYield,
      avgCapacity,
      disruptions,
      dataCount: data.length
    });

    return {
      totalRevenue,
      avgYield,
      avgCapacity,
      disruptions
    };
  }

  private getWeekKey(date: Date): string {
    const year = date.getFullYear();
    const week = this.getWeekNumber(date);
    return `${year}-W${week.toString().padStart(2, '0')}`;
  }

  private getWeekNumber(date: Date): number {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil((((d.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  }

  // Observable for reactive data access
  getDataObservable(): Observable<TFTLCDData | null> {
    return this.dataCache$.asObservable();
  }

  // Clear cache to force reload
  clearCache(): void {
    this.dataCache$.next(null);
    this.lastLoadTime = 0;
  }
}