// src/services/dataLoader.js
import Papa from 'papaparse';

export class DataLoaderService {
  constructor() {
    this.cache = new Map();
    this.isLoading = new Map();
  }

  /**
   * Load CSV data from the data folder
   * @param {string} filename - Name of the CSV file (without path)
   * @returns {Promise<Array>} Parsed CSV data
   */
  async loadCSV(filename) {
    // Check cache first
    if (this.cache.has(filename)) {
      return this.cache.get(filename);
    }

    // Check if already loading
    if (this.isLoading.has(filename)) {
      return this.isLoading.get(filename);
    }

    // Create loading promise
    const loadingPromise = this._loadCSVFile(filename);
    this.isLoading.set(filename, loadingPromise);

    try {
      const data = await loadingPromise;
      this.cache.set(filename, data);
      this.isLoading.delete(filename);
      return data;
    } catch (error) {
      this.isLoading.delete(filename);
      throw error;
    }
  }

  /**
   * Internal method to load and parse CSV file
   * @param {string} filename 
   * @returns {Promise<Array>}
   */
  async _loadCSVFile(filename) {
    try {
      // Construct the path to the CSV file in the data folder
      const dataPath = `/data/${filename}`;
      
      const response = await fetch(dataPath);
      if (!response.ok) {
        throw new Error(`Failed to load ${filename}: ${response.status} ${response.statusText}`);
      }

      const csvText = await response.text();
      
      return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          delimitersToGuess: [',', '\t', '|', ';'],
          transformHeader: (header) => {
            // Clean up header names
            return header.trim().replace(/\s+/g, '_').toLowerCase();
          },
          transform: (value, header) => {
            // Handle common data transformations
            if (typeof value === 'string') {
              value = value.trim();
              
              // Convert date strings to Date objects
              if (header.includes('date') && value) {
                const date = new Date(value);
                return isNaN(date.getTime()) ? value : date;
              }
              
              // Convert percentage strings to numbers
              if (value.includes('%')) {
                const numValue = parseFloat(value.replace('%', ''));
                return isNaN(numValue) ? value : numValue / 100;
              }
              
              // Convert currency strings to numbers
              if (value.startsWith('$')) {
                const numValue = parseFloat(value.replace(/[$,]/g, ''));
                return isNaN(numValue) ? value : numValue;
              }
            }
            
            return value;
          },
          complete: (results) => {
            if (results.errors.length > 0) {
              console.warn(`Parsing warnings for ${filename}:`, results.errors);
            }
            
            // Filter out rows with all null/undefined values
            const cleanData = results.data.filter(row => 
              Object.values(row).some(value => value !== null && value !== undefined && value !== '')
            );
            
            console.log(`Loaded ${filename}: ${cleanData.length} records`);
            resolve(cleanData);
          },
          error: (error) => {
            reject(new Error(`CSV parsing error for ${filename}: ${error.message}`));
          }
        });
      });
    } catch (error) {
      throw new Error(`Failed to load CSV file ${filename}: ${error.message}`);
    }
  }

  /**
   * Load all TFT-LCD data files
   * @returns {Promise<Object>} Object containing all loaded datasets
   */
  async loadAllTFTLCDData() {
    try {
      const [mainData, componentData, marketData] = await Promise.all([
        this.loadCSV('tft_lcd_main_data.csv'),
        this.loadCSV('tft_lcd_component_data.csv'),
        this.loadCSV('tft_lcd_market_data.csv')
      ]);

      return {
        mainData,
        componentData,
        marketData,
        summary: {
          mainRecords: mainData.length,
          componentRecords: componentData.length,
          marketRecords: marketData.length,
          dateRange: this._getDateRange(mainData),
          panelSizes: this._getUniqueValues(mainData, 'panel_size'),
          plants: this._getUniqueValues(mainData, 'plant'),
          marketSegments: this._getUniqueValues(mainData, 'market_segment')
        }
      };
    } catch (error) {
      console.error('Error loading TFT-LCD data:', error);
      throw error;
    }
  }

  /**
   * Get date range from data
   * @param {Array} data 
   * @returns {Object}
   */
  _getDateRange(data) {
    const dates = data
      .map(row => row.date)
      .filter(date => date instanceof Date)
      .sort((a, b) => a - b);
    
    return {
      start: dates[0],
      end: dates[dates.length - 1],
      totalWeeks: dates.length
    };
  }

  /**
   * Get unique values from a column
   * @param {Array} data 
   * @param {string} column 
   * @returns {Array}
   */
  _getUniqueValues(data, column) {
    return [...new Set(data.map(row => row[column]))].filter(value => 
      value !== null && value !== undefined
    ).sort();
  }

  /**
   * Filter data by criteria
   * @param {Array} data 
   * @param {Object} filters 
   * @returns {Array}
   */
  filterData(data, filters = {}) {
    return data.filter(row => {
      for (const [key, value] of Object.entries(filters)) {
        if (value !== 'all' && value !== null && value !== undefined) {
          if (row[key] !== value) {
            return false;
          }
        }
      }
      return true;
    });
  }

  /**
   * Aggregate data by specified grouping
   * @param {Array} data 
   * @param {string|Array} groupBy 
   * @param {Object} aggregations 
   * @returns {Array}
   */
  aggregateData(data, groupBy, aggregations = {}) {
    const groups = new Map();
    
    data.forEach(row => {
      const key = Array.isArray(groupBy) 
        ? groupBy.map(field => row[field]).join('|')
        : row[groupBy];
      
      if (!groups.has(key)) {
        const groupData = Array.isArray(groupBy)
          ? groupBy.reduce((obj, field, index) => {
              obj[field] = key.split('|')[index];
              return obj;
            }, {})
          : { [groupBy]: key };
        
        groups.set(key, { 
          ...groupData,
          count: 0,
          sum: {},
          avg: {},
          min: {},
          max: {}
        });
      }
      
      const group = groups.get(key);
      group.count++;
      
      // Calculate aggregations
      for (const [field, operations] of Object.entries(aggregations)) {
        const value = parseFloat(row[field]);
        if (!isNaN(value)) {
          if (operations.includes('sum')) {
            group.sum[field] = (group.sum[field] || 0) + value;
          }
          if (operations.includes('min')) {
            group.min[field] = Math.min(group.min[field] || Infinity, value);
          }
          if (operations.includes('max')) {
            group.max[field] = Math.max(group.max[field] || -Infinity, value);
          }
        }
      }
    });
    
    // Calculate averages
    const result = Array.from(groups.values()).map(group => {
      for (const [field, operations] of Object.entries(aggregations)) {
        if (operations.includes('avg') && group.sum[field] !== undefined) {
          group.avg[field] = group.sum[field] / group.count;
        }
      }
      return group;
    });
    
    return result;
  }

  /**
   * Generate time series data
   * @param {Array} data 
   * @param {string} dateField 
   * @param {Array} metrics 
   * @param {Object} filters 
   * @returns {Array}
   */
  generateTimeSeries(data, dateField = 'date', metrics = [], filters = {}) {
    const filteredData = this.filterData(data, filters);
    
    const timeGroups = this.aggregateData(
      filteredData,
      dateField,
      metrics.reduce((acc, metric) => {
        acc[metric] = ['sum', 'avg', 'min', 'max'];
        return acc;
      }, {})
    );
    
    return timeGroups.sort((a, b) => {
      const dateA = new Date(a[dateField]);
      const dateB = new Date(b[dateField]);
      return dateA - dateB;
    });
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
  }

  /**
   * Get cache status
   * @returns {Object}
   */
  getCacheStatus() {
    return {
      cachedFiles: Array.from(this.cache.keys()),
      loadingFiles: Array.from(this.isLoading.keys()),
      cacheSize: this.cache.size
    };
  }
}

// Create singleton instance
export const dataLoader = new DataLoaderService();

// Export convenience functions
export const loadTFTLCDData = () => dataLoader.loadAllTFTLCDData();
export const loadCSV = (filename) => dataLoader.loadCSV(filename);
export const filterData = (data, filters) => dataLoader.filterData(data, filters);
export const aggregateData = (data, groupBy, aggregations) => dataLoader.aggregateData(data, groupBy, aggregations);
export const generateTimeSeries = (data, dateField, metrics, filters) => dataLoader.generateTimeSeries(data, dateField, metrics, filters);