# TFT-LCD Synthetic Dataset for Multi-Objective Optimization Research

## Dataset Overview

This comprehensive synthetic dataset simulates 3 years (2022-2024) of TFT-LCD manufacturing operations, designed specifically for testing multi-objective optimization frameworks under uncertainty. The dataset reflects realistic industry characteristics and operational complexities.

## Dataset Structure

### 1. Main Production Dataset (`tft_lcd_main_data.csv`)

**Dimensions:** ~62,400 records (156 weeks × 5 panel sizes × 4 plants × 4 market segments)

**Time Period:** January 2022 - December 2024 (weekly frequency)

#### Key Variables:

**Temporal Dimensions:**
- `Date`: Week ending date
- `Year`, `Quarter`, `Month`, `Week`: Time period identifiers

**Product Dimensions:**
- `Panel_Size`: 32", 43", 55", 65", 75" displays
- `Market_Segment`: TV, Monitor, Laptop, Tablet
- `Plant`: Plant_TW_Taichung, Plant_KR_Paju, Plant_CN_Hefei, Plant_CN_Guangzhou
- `Region`: Taiwan, South Korea, China

**Demand and Sales:**
- `Forecasted_Demand`: AI-generated demand forecast (units)
- `Actual_Sales`: Realized sales (units)
- `Demand_Forecast_Accuracy`: Forecast accuracy ratio (0-1)

**Production Metrics:**
- `Planned_Production`: Production plan (units)
- `Actual_Production`: Actual production output (units)
- `Capacity_Utilization`: Plant capacity usage (0-1)
- `Production_Yield`: First-pass yield rate (0.75-0.98)
- `Good_Units_Produced`: Quality-passed units
- `Defective_Units`: Failed quality units

**Inventory Management:**
- `Beginning_Inventory`: Starting inventory (units)
- `Ending_Inventory`: Closing inventory (units)
- `Inventory_Turns`: Inventory turnover ratio
- `Stockout_Risk`: Risk of stock shortage (0-1)

**Financial Metrics:**
- `Unit_Selling_Price`: Revenue per unit ($)
- `Unit_Production_Cost`: Cost per unit ($)
- `Unit_Margin`: Profit per unit ($)
- `Revenue`: Total sales revenue ($)
- `Production_Cost_Total`: Total production costs ($)
- `Gross_Profit`: Revenue minus costs ($)
- `Inventory_Holding_Cost`: Inventory carrying costs ($)

**Quality and Service:**
- `Defect_Rate`: Production defect rate (0-0.25)
- `Customer_Complaints`: Number of complaints
- `On_Time_Delivery`: Delivery performance (0.85-0.98)
- `First_Pass_Yield`: Quality yield rate

**Supply Chain:**
- `Supply_Disruptions`: Binary disruption indicator
- `Average_Lead_Time`: Component lead times (days)

**External Factors:**
- `COVID_Impact_Factor`: Pandemic impact (0.6-1.1)
- `Economic_Cycle_Factor`: Economic conditions (0.85-1.05)
- `Technology_Trend_Factor`: Technology adoption (1.0-1.08)
- `Seasonal_Factor`: Seasonal demand patterns (0.8-1.3)

### 2. Component Pricing Dataset (`tft_lcd_component_data.csv`)

**Dimensions:** ~4,680 records (156 weeks × 6 components)

**Components:**
- Glass_Substrate (14-day lead time, 15% price volatility)
- Color_Filter (21-day lead time, 20% price volatility)
- Polarizer (28-day lead time, 25% price volatility)
- Driver_IC (35-day lead time, 30% price volatility)
- Backlight_Unit (18-day lead time, 18% price volatility)
- PCB_Assembly (12-day lead time, 12% price volatility)

**Variables:**
- `Date`: Week ending date
- `Component`: Component name
- `Price_Index`: Price index (base 100)
- `Lead_Time_Days`: Actual lead time with variance
- `Supply_Availability`: Availability rate (0.85-1.0)
- `Critical_Component`: Boolean criticality flag
- `Disruption_Event`: Supply disruption indicator

### 3. Market Indicators Dataset (`tft_lcd_market_data.csv`)

**Dimensions:** 156 records (weekly)

**Economic Indicators:**
- `GDP_Growth_Rate`: Quarterly GDP growth (%)
- `Inflation_Rate`: Consumer price inflation (%)
- `USD_CNY_Exchange_Rate`: Currency exchange rate
- `Consumer_Confidence_Index`: Market confidence (0-100)

**Industry Metrics:**
- `TV_Global_Shipments`: Global TV unit shipments
- `Monitor_Demand_Index`: Monitor demand index (base 100)
- `OLED_Market_Share`: OLED technology adoption (%)
- `MicroLED_Adoption_Rate`: Emerging technology rate (%)
- `Display_Technology_Index`: Technology advancement index
- `Energy_Cost_Index`: Manufacturing energy costs

## Industry-Realistic Characteristics

### Market Dynamics:
- **Seasonal Patterns**: Q4 peak for TV (holiday season), Q3 peak for monitors (back-to-school)
- **Size Trends**: Gradual shift toward larger panels (65", 75" growing 5-8% annually)
- **Price Deflation**: 3% annual price decline (typical for display industry)
- **Regional Variations**: China highest demand, Taiwan most efficient production

### Manufacturing Constraints:
- **Capacity Limits**: Larger panels require more production capacity
- **Yield Rates**: 85-95% typical yields, varying by plant efficiency
- **Quality Focus**: Defect tracking and customer complaint metrics
- **Supply Chain**: Component lead times 12-35 days with disruption risks

### Uncertainty Sources:
- **Demand Volatility**: 15% log-normal demand uncertainty
- **Supply Disruptions**: 5-8% weekly probability, higher during Chinese New Year
- **Yield Variations**: ±5% yield fluctuations
- **Price Volatility**: Component prices vary 12-30% based on market conditions
- **External Shocks**: COVID impact, economic cycles, technology disruptions

## Sample Data Records

### Main Dataset Sample:
```
Date: 2023-01-02, Week: 1
Panel: 55" TV at Plant_TW_Taichung
Demand: 1,847 | Production: 2,156 | Yield: 92.1%
Sales: 1,923 | Revenue: $498,452
Inventory: Begin 623, End 865
Price: $259.21 | Cost: $195.34 | Margin: $63.87
Supply Disruptions: 0 | On-Time Delivery: 94.2%
```

### Component Dataset Sample:
```
Date: 2023-01-02
Component: Driver_IC
Price Index: 187.45 | Lead Time: 31 days
Supply Availability: 91.2% | Disruption: No
Quality Score: 96.8%
```

## Use Cases for Optimization Research

### Multi-Objective Optimization Targets:
1. **Cost Minimization**: Reduce production and inventory costs
2. **Service Level Maximization**: Improve on-time delivery and reduce stockouts
3. **Quality Optimization**: Maximize yield rates and minimize defects
4. **Resilience Enhancement**: Build robustness against supply disruptions

### Uncertainty Modeling:
- **Demand Uncertainty**: Forecast errors, market volatility
- **Supply Uncertainty**: Component delays, quality variations
- **Operational Uncertainty**: Yield fluctuations, capacity constraints
- **External Uncertainty**: Economic cycles, technology shifts

### Optimization Scenarios:
- **Production Planning**: Weekly production scheduling across plants
- **Inventory Management**: Safety stock optimization by product-location
- **Capacity Allocation**: Plant capacity assignment across product lines
- **Supply Chain Design**: Supplier selection and contract optimization

## Data Quality and Validation

### Realistic Relationships:
- Larger panels → Higher prices, lower yields, more capacity usage
- Better plants → Higher efficiency, better yields, lower costs
- Higher demand → More production, potential capacity constraints
- Supply disruptions → Higher costs, delivery delays

### Correlation Structure:
- Demand forecast accuracy correlates with reduced stockout risk
- Higher yields correlate with lower unit costs and better margins
- Capacity utilization correlates with production efficiency
- External factors create realistic demand patterns

### Validation Benchmarks:
- Industry yield rates: 85-95% (✓ Dataset: 80-98%)
- Price deflation: ~3% annually (✓ Dataset: 3% deflation)
- Seasonal patterns: Q4 TV peak (✓ Dataset: 30% Q4 increase)
- Supply disruption frequency: 5-10% (✓ Dataset: 5-8%)

## Usage Instructions

### Loading the Dataset:
```python
import pandas as pd

# Load main production data
main_df = pd.read_csv('tft_lcd_main_data.csv')
main_df['Date'] = pd.to_datetime(main_df['Date'])

# Load component data
component_df = pd.read_csv('tft_lcd_component_data.csv')
component_df['Date'] = pd.to_datetime(component_df['Date'])

# Load market data
market_df = pd.read_csv('tft_lcd_market_data.csv')
market_df['Date'] = pd.to_datetime(market_df['Date'])
```

### Key Analysis Dimensions:
- **Time Series**: Weekly trends, seasonal patterns, year-over-year growth
- **Cross-Sectional**: Plant performance, product profitability, market segments
- **Panel Analysis**: Time-series cross-sectional analysis across all dimensions

### Optimization Framework Integration:
- **Objective Functions**: Revenue, costs, service levels, quality metrics
- **Decision Variables**: Production quantities, inventory levels, capacity allocation
- **Constraints**: Capacity limits, demand requirements, inventory bounds
- **Uncertainty Parameters**: Demand, yield, lead times, disruptions

This synthetic dataset provides a comprehensive foundation for testing and validating multi-objective optimization frameworks in realistic manufacturing environments, supporting both academic research and practical industry applications.