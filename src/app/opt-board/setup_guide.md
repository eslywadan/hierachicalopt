# Setup Guide: Integrating CSV Data with Analysis Tools

## 📁 Required File Structure

To connect your TFT-LCD synthetic data with the analysis tools, organize your project structure as follows:

```
hierachicalopt/
├── src/
│   ├── analysis/
│   │   └── opt-board/
│   │       ├── src/
│   │       │   ├── components/
│   │       │   │   ├── TFTLCDDashboard.js
│   │       │   │   ├── TFTLCDDashboard.css
│   │       │   │   ├── OptimizationAnalysisTools.js
│   │       │   │   └── OptimizationAnalysisTools.css
│   │       │   ├── hooks/
│   │       │   │   ├── useTFTLCDData.js
│   │       │   │   └── useOptimizationData.js
│   │       │   ├── services/
│   │       │   │   └── dataLoader.js
│   │       │   ├── App.js
│   │       │   ├── App.css
│   │       │   └── index.js
│   │       ├── public/
│   │       │   ├── data/              # ← CSV files go here
│   │       │   │   ├── tft_lcd_main_data.csv
│   │       │   │   ├── tft_lcd_component_data.csv
│   │       │   │   └── tft_lcd_market_data.csv
│   │       │   └── index.html
│   │       └── package.json
│   └── data/                          # ← Your original CSV location
│       ├── tft_lcd_main_data.csv
│       ├── tft_lcd_component_data.csv
│       └── tft_lcd_market_data.csv
└── README.md
```

## 🔧 Installation Steps

### 1. Navigate to the Analysis Tool Directory
```bash
cd src/analysis/opt-board
```

### 2. Install Dependencies
```bash
npm install
npm install papaparse recharts lucide-react
```

### 3. Copy CSV Files to Public Directory
```bash
# Create the data directory in public folder
mkdir -p public/data

# Copy your CSV files from the main data folder
cp ../../../data/tft_lcd_*.csv public/data/
```

Or manually copy the files:
- Copy `src/data/tft_lcd_main_data.csv` → `src/analysis/opt-board/public/data/tft_lcd_main_data.csv`
- Copy `src/data/tft_lcd_component_data.csv` → `src/analysis/opt-board/public/data/tft_lcd_component_data.csv`
- Copy `src/data/tft_lcd_market_data.csv` → `src/analysis/opt-board/public/data/tft_lcd_market_data.csv`

### 4. Verify File Structure
Check that your files are in the correct location:
```bash
ls -la public/data/
# Should show:
# tft_lcd_main_data.csv
# tft_lcd_component_data.csv
# tft_lcd_market_data.csv
```

### 5. Update Package.json (if needed)
Ensure your `package.json` includes all required dependencies:
```json
{
  "name": "tft-lcd-optimization-dashboard",
  "version": "0.1.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.8.0",
    "lucide-react": "^0.263.1",
    "papaparse": "^5.4.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }
}
```

## 🚀 Running the Application

### Start the Development Server
```bash
npm start
```

The application will:
1. Automatically load CSV data from `public/data/` folder
2. Parse and process the data using PapaParse
3. Display real manufacturing data in the dashboards
4. Generate optimization scenarios based on actual performance metrics

## 📊 CSV File Requirements

### Expected CSV Format

**tft_lcd_main_data.csv** should contain columns like:
```csv
Date,Panel_Size,Plant,Market_Segment,Forecasted_Demand,Actual_Production,Production_Yield,Revenue,Unit_Selling_Price,Unit_Production_Cost,Supply_Disruptions,On_Time_Delivery,Capacity_Utilization
2023-01-02,43,Plant_TW_Taichung,TV,1500,1450,0.92,261000,180,126,0,0.94,0.87
```

**tft_lcd_component_data.csv** should contain:
```csv
Date,Component,Price_Index,Lead_Time_Days,Supply_Availability,Quality_Score
2023-01-02,Glass_Substrate,98.5,14,0.95,0.98
```

**tft_lcd_market_data.csv** should contain:
```csv
Date,GDP_Growth_Rate,Inflation_Rate,TV_Global_Shipments,Monitor_Demand_Index
2023-01-02,2.1,3.2,38500000,105.2
```

### Column Name Handling
The data loader automatically:
- Converts column names to lowercase with underscores
- Trims whitespace from headers
- Handles common data types (dates, percentages, currency)

## 🔍 Data Validation

### Check Data Loading
1. Open browser developer tools (F12)
2. Look for console messages like:
   ```
   Loaded tft_lcd_main_data.csv: 15840 records
   Loaded tft_lcd_component_data.csv: 936 records
   Loaded tft_lcd_market_data.csv: 156 records
   ```

### Verify Data Integration
- **Dashboard Tab**: Should show real data trends and metrics
- **Optimization Tab**: Should display "Based on real TFT-LCD manufacturing data"
- **Filter Controls**: Should populate with actual plant names, panel sizes, etc.

## 🛠️ Troubleshooting

### Common Issues

**1. CSV Files Not Loading**
```
Error: Failed to load tft_lcd_main_data.csv: 404 Not Found
```
**Solution**: Ensure CSV files are in `public/data/` directory, not `src/data/`

**2. Parsing Errors**
```
CSV parsing error: Unexpected delimiter
```
**Solution**: Check CSV format and ensure consistent delimiters (commas)

**3. Missing Dependencies**
```
Module not found: Can't resolve 'papaparse'
```
**Solution**: Install missing packages:
```bash
npm install papaparse recharts lucide-react
```

**4. Data Not Displaying**
- Check browser console for error messages
- Verify CSV column names match expected format
- Ensure date columns are in recognizable format (YYYY-MM-DD)

### Debug Mode
Enable debug logging by adding to browser console:
```javascript
localStorage.setItem('debug', 'true');
```

## 📈 Features After Integration

### Real Data Dashboards
- **Interactive Filtering**: Filter by actual plant names, panel sizes, time ranges
- **KPI Cards**: Real revenue, yield, capacity utilization from your data
- **Time Series**: Actual weekly trends from your manufacturing data
- **Performance Analysis**: Plant-by-plant comparison using real metrics

### Optimization Analysis
- **Realistic Scenarios**: Optimization problems based on actual cost/quality ranges
- **Data-Driven Insights**: Recommendations based on real performance patterns
- **Validation**: Compare optimization results against historical performance

### Export Capabilities
- Charts can be exported for presentations
- Data summaries show actual record counts and date ranges
- Filtered views can be saved for different analysis scenarios

## 🎯 Next Steps

1. **Verify the setup** by running the application
2. **Explore the dashboards** with your real data
3. **Test filtering capabilities** across different dimensions
4. **Use optimization analysis** for research validation
5. **Export visualizations** for your thesis presentation

