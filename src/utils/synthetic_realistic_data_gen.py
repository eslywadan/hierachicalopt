import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class TFTLCDDataGenerator:
    def __init__(self):
        # Industry-realistic parameters
        self.panel_sizes = {
            32: {'market_share': 0.20, 'base_price': 120, 'complexity': 1.0},
            43: {'market_share': 0.25, 'base_price': 180, 'complexity': 1.2},
            55: {'market_share': 0.30, 'base_price': 280, 'complexity': 1.5},
            65: {'market_share': 0.15, 'base_price': 450, 'complexity': 2.0},
            75: {'market_share': 0.10, 'base_price': 650, 'complexity': 2.5}
        }
        
        # Manufacturing plants (representing major production regions)
        self.plants = {
            'Plant_TW_Taichung': {'capacity_factor': 1.2, 'efficiency': 0.92, 'region': 'Taiwan'},
            'Plant_KR_Paju': {'capacity_factor': 1.0, 'efficiency': 0.90, 'region': 'South Korea'},
            'Plant_CN_Hefei': {'capacity_factor': 1.1, 'efficiency': 0.88, 'region': 'China'},
            'Plant_CN_Guangzhou': {'capacity_factor': 0.9, 'efficiency': 0.87, 'region': 'China'},
        }
        
        # Supply chain components
        self.supply_components = {
            'Glass_Substrate': {'lead_time': 14, 'price_volatility': 0.15, 'critical': True},
            'Color_Filter': {'lead_time': 21, 'price_volatility': 0.20, 'critical': True},
            'Polarizer': {'lead_time': 28, 'price_volatility': 0.25, 'critical': True},
            'Driver_IC': {'lead_time': 35, 'price_volatility': 0.30, 'critical': True},
            'Backlight_Unit': {'lead_time': 18, 'price_volatility': 0.18, 'critical': False},
            'PCB_Assembly': {'lead_time': 12, 'price_volatility': 0.12, 'critical': False}
        }
        
        # Market segments
        self.market_segments = {
            'TV': {'share': 0.65, 'seasonality_peak': 'Q4', 'price_sensitivity': 0.8},
            'Monitor': {'share': 0.20, 'seasonality_peak': 'Q3', 'price_sensitivity': 0.6},
            'Laptop': {'share': 0.10, 'seasonality_peak': 'Q3', 'price_sensitivity': 0.7},
            'Tablet': {'share': 0.05, 'seasonality_peak': 'Q4', 'price_sensitivity': 0.9}
        }
        
    def generate_time_series_data(self, start_date='2022-01-01', end_date='2024-12-31', freq='W'):
        """Generate comprehensive time series data"""
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        data = []
        
        for date in dates:
            week_of_year = date.isocalendar()[1]
            month = date.month
            quarter = (month - 1) // 3 + 1
            year = date.year
            
            # Global demand multipliers
            covid_impact = self._covid_impact_factor(date)
            economic_cycle = self._economic_cycle_factor(date)
            tech_trend = self._technology_trend_factor(date)
            
            for size, size_info in self.panel_sizes.items():
                for segment, seg_info in self.market_segments.items():
                    for plant, plant_info in self.plants.items():
                        
                        # Base demand calculation
                        base_weekly_demand = (
                            size_info['market_share'] * 
                            seg_info['share'] * 
                            50000  # Base weekly units across all products
                        )
                        
                        # Apply various factors
                        seasonal_factor = self._seasonal_factor(quarter, segment)
                        regional_factor = self._regional_demand_factor(plant_info['region'], date)
                        size_trend = self._size_trend_factor(size, year)
                        
                        # Final demand with uncertainty
                        demand = (
                            base_weekly_demand * 
                            covid_impact * 
                            economic_cycle * 
                            tech_trend * 
                            seasonal_factor * 
                            regional_factor * 
                            size_trend * 
                            np.random.lognormal(0, 0.15)  # Demand uncertainty
                        )
                        
                        # Production planning
                        planned_production = demand * np.random.uniform(1.05, 1.25)  # Safety stock
                        
                        # Capacity constraints
                        max_capacity = (
                            plant_info['capacity_factor'] * 
                            15000 *  # Weekly capacity per plant-size combination
                            (2.0 / size_info['complexity'])  # Larger panels take more capacity
                        )
                        
                        actual_production = min(planned_production, max_capacity)
                        
                        # Yield and quality
                        base_yield = plant_info['efficiency']
                        yield_variation = np.random.normal(0, 0.05)  # 5% std dev
                        actual_yield = np.clip(base_yield + yield_variation, 0.75, 0.98)
                        
                        good_units = actual_production * actual_yield
                        
                        # Inventory dynamics
                        if len(data) > 0:
                            # Find previous week's inventory for same product-plant combo
                            prev_inventory = 1000  # Default starting inventory
                            for prev_row in reversed(data[-50:]):  # Search recent history
                                if (prev_row['Panel_Size'] == size and 
                                    prev_row['Plant'] == plant and 
                                    prev_row['Market_Segment'] == segment):
                                    prev_inventory = prev_row['Ending_Inventory']
                                    break
                        else:
                            prev_inventory = 1000
                        
                        # Inventory calculation
                        beginning_inventory = prev_inventory
                        total_available = beginning_inventory + good_units
                        actual_sales = min(demand * np.random.uniform(0.9, 1.1), total_available)
                        ending_inventory = total_available - actual_sales
                        
                        # Supply chain metrics
                        supply_disruptions = self._generate_supply_disruptions(date, size)
                        lead_time_variance = np.random.uniform(0.8, 1.3)  # Lead time uncertainty
                        
                        # Financial metrics
                        base_price = size_info['base_price']
                        market_price_factor = self._market_price_factor(date, segment, size)
                        unit_price = base_price * market_price_factor * np.random.uniform(0.95, 1.05)
                        
                        production_cost = self._calculate_production_cost(size, plant_info, actual_yield)
                        
                        # Quality metrics
                        defect_rate = (1 - actual_yield)
                        customer_complaints = max(0, np.random.poisson(defect_rate * actual_sales * 0.001))
                        
                        # Create record
                        record = {
                            'Date': date,
                            'Year': year,
                            'Quarter': quarter,
                            'Month': month,
                            'Week': week_of_year,
                            'Panel_Size': size,
                            'Market_Segment': segment,
                            'Plant': plant,
                            'Region': plant_info['region'],
                            
                            # Demand and Sales
                            'Forecasted_Demand': round(demand),
                            'Actual_Sales': round(actual_sales),
                            'Demand_Forecast_Accuracy': min(1.0, actual_sales / demand) if demand > 0 else 1.0,
                            
                            # Production
                            'Planned_Production': round(planned_production),
                            'Actual_Production': round(actual_production),
                            'Capacity_Utilization': actual_production / max_capacity if max_capacity > 0 else 0,
                            'Production_Yield': actual_yield,
                            'Good_Units_Produced': round(good_units),
                            'Defective_Units': round(actual_production - good_units),
                            
                            # Inventory
                            'Beginning_Inventory': round(beginning_inventory),
                            'Ending_Inventory': round(ending_inventory),
                            'Inventory_Turns': (actual_sales / ((beginning_inventory + ending_inventory) / 2)) if beginning_inventory + ending_inventory > 0 else 0,
                            'Stockout_Risk': max(0, (demand - total_available) / demand) if demand > 0 else 0,
                            
                            # Supply Chain
                            'Supply_Disruptions': supply_disruptions,
                            'Average_Lead_Time': np.mean([comp['lead_time'] * lead_time_variance for comp in self.supply_components.values()]),
                            'On_Time_Delivery': np.random.uniform(0.85, 0.98) * (1 - supply_disruptions * 0.3),
                            
                            # Financial
                            'Unit_Selling_Price': round(unit_price, 2),
                            'Unit_Production_Cost': round(production_cost, 2),
                            'Unit_Margin': round(unit_price - production_cost, 2),
                            'Revenue': round(actual_sales * unit_price, 2),
                            'Production_Cost_Total': round(actual_production * production_cost, 2),
                            'Inventory_Holding_Cost': round(ending_inventory * production_cost * 0.02, 2),  # 2% weekly holding cost
                            
                            # Quality
                            'Defect_Rate': round(defect_rate, 4),
                            'Customer_Complaints': customer_complaints,
                            'First_Pass_Yield': actual_yield,
                            
                            # External Factors
                            'COVID_Impact_Factor': covid_impact,
                            'Economic_Cycle_Factor': economic_cycle,
                            'Technology_Trend_Factor': tech_trend,
                            'Seasonal_Factor': seasonal_factor,
                        }
                        
                        data.append(record)
        
        return pd.DataFrame(data)
    
    def _seasonal_factor(self, quarter, segment):
        """Calculate seasonal demand factors"""
        seasonal_patterns = {
            'TV': [0.8, 0.9, 1.0, 1.3],  # Q4 peak (holiday season)
            'Monitor': [0.9, 0.9, 1.2, 1.0],  # Q3 peak (back-to-school)
            'Laptop': [0.9, 0.8, 1.3, 1.0],  # Q3 peak (back-to-school)
            'Tablet': [0.8, 0.9, 1.1, 1.2]   # Q4 peak (holidays)
        }
        return seasonal_patterns[segment][quarter-1]
    
    def _covid_impact_factor(self, date):
        """Model COVID-19 impact on demand"""
        if date < datetime(2020, 3, 1):
            return 1.0
        elif date < datetime(2020, 6, 1):
            return 0.6  # Severe impact
        elif date < datetime(2021, 1, 1):
            return 0.8  # Recovery phase
        elif date < datetime(2022, 1, 1):
            return 1.1  # Pent-up demand
        else:
            return 1.0  # Normal
    
    def _economic_cycle_factor(self, date):
        """Model economic cycles"""
        # Simplified economic cycle
        years_since_2020 = (date.year - 2020) + date.month / 12.0
        cycle = 0.95 + 0.1 * np.sin(2 * np.pi * years_since_2020 / 3.0)  # 3-year cycle
        return cycle
    
    def _technology_trend_factor(self, date):
        """Model technology adoption trends"""
        years_since_2020 = (date.year - 2020) + date.month / 12.0
        # Gradual technology improvement driving demand
        return 1.0 + 0.02 * years_since_2020
    
    def _regional_demand_factor(self, region, date):
        """Regional demand variations"""
        base_factors = {
            'Taiwan': 1.0,
            'South Korea': 0.9,
            'China': 1.2
        }
        
        # Add some regional economic variations
        regional_cycles = {
            'Taiwan': 0.05 * np.sin(2 * np.pi * date.month / 12),
            'South Korea': 0.03 * np.cos(2 * np.pi * date.month / 12),
            'China': 0.08 * np.sin(2 * np.pi * date.month / 6)  # Faster cycle
        }
        
        return base_factors[region] * (1 + regional_cycles[region])
    
    def _size_trend_factor(self, size, year):
        """Trend towards larger panels over time"""
        size_trends = {
            32: 0.95 ** (year - 2022),  # Declining
            43: 0.98 ** (year - 2022),  # Slight decline
            55: 1.02 ** (year - 2022),  # Growing
            65: 1.05 ** (year - 2022),  # Strong growth
            75: 1.08 ** (year - 2022)   # Very strong growth
        }
        return size_trends[size]
    
    def _market_price_factor(self, date, segment, size):
        """Calculate market-driven price variations"""
        base_factor = 1.0
        
        # Larger panels command premium
        size_premium = {32: 0.95, 43: 1.0, 55: 1.05, 65: 1.1, 75: 1.15}[size]
        
        # Market segment pricing
        segment_factor = {
            'TV': 0.9,      # Commodity pricing
            'Monitor': 1.1,  # Premium for professional use
            'Laptop': 1.05, # Moderate premium
            'Tablet': 0.95  # Competitive market
        }[segment]
        
        # Temporal price trends (deflation in display industry)
        years_since_2022 = (date.year - 2022) + date.month / 12.0
        price_deflation = 0.97 ** years_since_2022  # 3% annual price decline
        
        # Market volatility
        volatility = np.random.uniform(0.95, 1.05)
        
        return base_factor * size_premium * segment_factor * price_deflation * volatility
    
    def _calculate_production_cost(self, size, plant_info, yield_rate):
        """Calculate production cost per unit"""
        # Base cost components
        material_cost = size * 1.2  # Proportional to panel size
        labor_cost = 15 + size * 0.3  # Base + size-dependent
        overhead_cost = 25  # Fixed overhead
        
        # Plant efficiency impact
        efficiency_factor = 2.0 - plant_info['efficiency']  # Better plants have lower costs
        
        # Yield impact (lower yield = higher cost per good unit)
        yield_factor = 1.0 / max(yield_rate, 0.5)  # Avoid division by very small numbers
        
        total_cost = (material_cost + labor_cost + overhead_cost) * efficiency_factor * yield_factor
        
        return total_cost
    
    def _generate_supply_disruptions(self, date, size):
        """Generate supply chain disruption events"""
        # Base disruption probability
        base_prob = 0.05  # 5% weekly probability
        
        # Size complexity factor
        complexity_factor = {32: 1.0, 43: 1.1, 55: 1.2, 65: 1.4, 75: 1.6}[size]
        
        # Seasonal factors (Chinese New Year, summer holidays)
        seasonal_disruption = 1.0
        if date.month in [1, 2]:  # Chinese New Year period
            seasonal_disruption = 2.0
        elif date.month in [7, 8]:  # Summer maintenance
            seasonal_disruption = 1.5
        
        # Random disruption events
        disruption_prob = base_prob * complexity_factor * seasonal_disruption
        return 1 if np.random.random() < disruption_prob else 0
    
    def generate_component_price_data(self, main_df):
        """Generate component pricing data"""
        component_data = []
        dates = main_df['Date'].unique()
        
        for date in dates:
            for component, comp_info in self.supply_components.items():
                # Base price (normalized)
                base_price = 100
                
                # Trend (components generally get cheaper over time)
                years_since_2022 = (pd.to_datetime(date).year - 2022) + pd.to_datetime(date).month / 12.0
                trend_factor = 0.95 ** years_since_2022
                
                # Volatility based on component characteristics
                volatility = np.random.normal(1.0, comp_info['price_volatility'])
                
                # Supply disruption impact
                disruption_impact = 1.0
                if np.random.random() < 0.1:  # 10% chance of supply issue
                    disruption_impact = np.random.uniform(1.2, 2.0)
                
                price = base_price * trend_factor * volatility * disruption_impact
                
                component_data.append({
                    'Date': date,
                    'Component': component,
                    'Price_Index': round(price, 2),
                    'Lead_Time_Days': comp_info['lead_time'] * np.random.uniform(0.8, 1.3),
                    'Supply_Availability': np.random.uniform(0.85, 1.0),
                    'Critical_Component': comp_info['critical'],
                    'Disruption_Event': disruption_impact > 1.1
                })
        
        return pd.DataFrame(component_data)
    
    def save_datasets(self, base_filename='tft_lcd_synthetic'):
        """Generate and save all datasets"""
        print("Generating main production dataset...")
        main_df = self.generate_time_series_data()
        
        print("Generating component pricing dataset...")
        component_df = self.generate_component_price_data(main_df)
        
        # Save datasets
        main_df.to_csv(f'{base_filename}_main.csv', index=False)
        component_df.to_csv(f'{base_filename}_components.csv', index=False)
        
        print(f"\nDatasets saved:")
        print(f"- {base_filename}_main.csv: {len(main_df)} records")
        print(f"- {base_filename}_components.csv: {len(component_df)} records")
        
        # Generate summary statistics
        self.generate_summary_report(main_df, component_df, base_filename)
        
        return main_df, component_df
    
    def generate_summary_report(self, main_df, component_df, base_filename):
        """Generate summary statistics and visualizations"""
        
        # Summary statistics
        summary_stats = {
            'Dataset Overview': {
                'Time Period': f"{main_df['Date'].min()} to {main_df['Date'].max()}",
                'Total Records': len(main_df),
                'Panel Sizes': list(main_df['Panel_Size'].unique()),
                'Manufacturing Plants': list(main_df['Plant'].unique()),
                'Market Segments': list(main_df['Market_Segment'].unique()),
            },
            'Production Metrics': {
                'Average Weekly Production': f"{main_df['Actual_Production'].mean():.0f} units",
                'Average Capacity Utilization': f"{main_df['Capacity_Utilization'].mean():.1%}",
                'Average Yield Rate': f"{main_df['Production_Yield'].mean():.1%}",
                'Average Inventory Turns': f"{main_df['Inventory_Turns'].mean():.1f}",
            },
            'Financial Metrics': {
                'Average Unit Price': f"${main_df['Unit_Selling_Price'].mean():.2f}",
                'Average Unit Margin': f"${main_df['Unit_Margin'].mean():.2f}",
                'Total Revenue (3 years)': f"${main_df['Revenue'].sum()/1e6:.1f}M",
            },
            'Quality Metrics': {
                'Average Defect Rate': f"{main_df['Defect_Rate'].mean():.2%}",
                'Average On-Time Delivery': f"{main_df['On_Time_Delivery'].mean():.1%}",
            }
        }
        
        # Save summary report
        with open(f'{base_filename}_summary.txt', 'w') as f:
            f.write("TFT-LCD Synthetic Dataset Summary Report\n")
            f.write("="*50 + "\n\n")
            
            for category, metrics in summary_stats.items():
                f.write(f"{category}:\n")
                f.write("-" * len(category) + "\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")
        
        print(f"- {base_filename}_summary.txt: Summary report")
        print("\nDataset generation complete!")

# Simplified dataset generation for immediate use
def create_sample_dataset():
    """Create a smaller sample dataset for demonstration"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    # Generate 12 weeks of data
    dates = pd.date_range('2023-01-02', '2023-03-27', freq='W')
    
    data = []
    
    for date in dates:
        week = date.isocalendar()[1] 
        month = date.month
        quarter = (month - 1) // 3 + 1
        
        for size in [32, 43, 55, 65]:
            for plant in ['Plant_TW_Taichung', 'Plant_KR_Paju', 'Plant_CN_Hefei']:
                for segment in ['TV', 'Monitor', 'Laptop']:
                    
                    # Base demand with factors
                    base_demand = 2000
                    size_factor = {32: 1.2, 43: 1.0, 55: 0.8, 65: 0.6}[size]
                    segment_factor = {'TV': 1.0, 'Monitor': 0.7, 'Laptop': 0.5}[segment]
                    plant_factor = {'Plant_TW_Taichung': 1.0, 'Plant_KR_Paju': 0.9, 'Plant_CN_Hefei': 1.1}[plant]
                    
                    # Seasonal effect
                    seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
                    
                    # Calculate demand
                    demand = (base_demand * size_factor * segment_factor * 
                             plant_factor * seasonal * np.random.uniform(0.8, 1.2))
                    
                    # Production planning
                    planned_production = demand * np.random.uniform(1.1, 1.25)
                    max_capacity = 3000
                    actual_production = min(planned_production, max_capacity)
                    
                    # Quality and yield
                    plant_efficiency = {'Plant_TW_Taichung': 0.92, 'Plant_KR_Paju': 0.90, 'Plant_CN_Hefei': 0.88}[plant]
                    yield_rate = plant_efficiency + np.random.normal(0, 0.03)
                    yield_rate = np.clip(yield_rate, 0.8, 0.98)
                    good_units = actual_production * yield_rate
                    
                    # Inventory
                    beginning_inventory = 500 + np.random.randint(-200, 300)
                    total_available = beginning_inventory + good_units
                    sales = min(demand * np.random.uniform(0.9, 1.05), total_available)
                    ending_inventory = total_available - sales
                    
                    # Pricing
                    base_price = {32: 120, 43: 180, 55: 280, 65: 450}[size]
                    price_factor = {'TV': 0.9, 'Monitor': 1.1, 'Laptop': 1.05}[segment]
                    unit_price = base_price * price_factor * np.random.uniform(0.95, 1.05)
                    
                    # Costs
                    material_cost = size * 1.2
                    labor_cost = 15 + size * 0.25
                    overhead = 25
                    efficiency_factor = 2.0 - plant_efficiency
                    production_cost = (material_cost + labor_cost + overhead) * efficiency_factor / yield_rate
                    
                    # Supply chain metrics
                    disruption = 1 if np.random.random() < 0.08 else 0
                    on_time_delivery = np.random.uniform(0.88, 0.97) * (1 - disruption * 0.25)
                    lead_time = 21 + np.random.randint(-5, 8)
                    
                    # Create record
                    record = {
                        'Date': date,
                        'Week': week,
                        'Month': month,
                        'Quarter': quarter,
                        'Panel_Size': size,
                        'Plant': plant,
                        'Market_Segment': segment,
                        'Forecasted_Demand': round(demand),
                        'Planned_Production': round(planned_production),
                        'Actual_Production': round(actual_production),
                        'Max_Capacity': max_capacity,
                        'Capacity_Utilization': round(actual_production / max_capacity, 3),
                        'Production_Yield': round(yield_rate, 3),
                        'Good_Units_Produced': round(good_units),
                        'Defective_Units': round(actual_production - good_units),
                        'Beginning_Inventory': round(beginning_inventory),
                        'Actual_Sales': round(sales),
                        'Ending_Inventory': round(ending_inventory),
                        'Inventory_Turns': round((sales / ((beginning_inventory + ending_inventory) / 2)) if beginning_inventory + ending_inventory > 0 else 0, 2),
                        'Unit_Selling_Price': round(unit_price, 2),
                        'Unit_Production_Cost': round(production_cost, 2),
                        'Unit_Margin': round(unit_price - production_cost, 2),
                        'Revenue': round(sales * unit_price),
                        'Gross_Profit': round(sales * (unit_price - production_cost)),
                        'Defect_Rate': round(1 - yield_rate, 4),
                        'On_Time_Delivery': round(on_time_delivery, 3),
                        'Supply_Disruptions': disruption,
                        'Lead_Time_Days': lead_time,
                        'Stockout_Risk': round(max(0, (demand - total_available) / demand) if demand > 0 else 0, 3)
                    }
                    
                    data.append(record)
    
    return pd.DataFrame(data)

def create_component_dataset():
    """Create component pricing and supply data"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-02', '2023-03-27', freq='W')
    components = [
        'Glass_Substrate', 'Color_Filter', 'Polarizer', 
        'Driver_IC', 'Backlight_Unit', 'PCB_Assembly'
    ]
    
    component_info = {
        'Glass_Substrate': {'base_price': 100, 'volatility': 0.15, 'lead_time': 14},
        'Color_Filter': {'base_price': 150, 'volatility': 0.20, 'lead_time': 21},
        'Polarizer': {'base_price': 120, 'volatility': 0.25, 'lead_time': 28},
        'Driver_IC': {'base_price': 200, 'volatility': 0.30, 'lead_time': 35},
        'Backlight_Unit': {'base_price': 80, 'volatility': 0.18, 'lead_time': 18},
        'PCB_Assembly': {'base_price': 60, 'volatility': 0.12, 'lead_time': 12}
    }
    
    data = []
    
    for date in dates:
        for component in components:
            info = component_info[component]
            
            # Price calculation
            base_price = info['base_price']
            trend_factor = 0.98 ** (date.month / 12)  # Slight price decline
            volatility = np.random.normal(1.0, info['volatility'])
            
            # Supply disruption impact
            disruption_impact = 1.0
            if np.random.random() < 0.1:  # 10% chance of supply issue
                disruption_impact = np.random.uniform(1.2, 1.8)
            
            price = base_price * trend_factor * volatility * disruption_impact
            
            record = {
                'Date': date,
                'Component': component,
                'Price_Index': round(price, 2),
                'Base_Lead_Time': info['lead_time'],
                'Actual_Lead_Time': round(info['lead_time'] * np.random.uniform(0.8, 1.3)),
                'Supply_Availability': round(np.random.uniform(0.85, 1.0), 3),
                'Quality_Score': round(np.random.uniform(0.9, 1.0), 3),
                'Supplier_Performance': round(np.random.uniform(0.85, 0.98), 3),
                'Disruption_Event': disruption_impact > 1.1
            }
            
            data.append(record)
    
    return pd.DataFrame(data)

def create_market_data():
    """Create market and economic indicators"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-02', '2023-03-27', freq='W')
    
    data = []
    
    for i, date in enumerate(dates):
        # Economic indicators
        gdp_growth = 2.5 + 0.5 * np.sin(2 * np.pi * i / 52) + np.random.normal(0, 0.2)
        inflation_rate = 3.0 + np.random.normal(0, 0.3)
        exchange_rate_usd_cny = 6.8 + np.random.normal(0, 0.1)
        
        # Industry indicators
        tv_shipments_global = 40000000 + np.random.randint(-2000000, 2000000)
        monitor_demand_index = 100 + np.random.uniform(-10, 10)
        
        # Technology trends
        oled_market_share = 0.15 + 0.01 * i / len(dates) + np.random.uniform(-0.01, 0.01)
        micro_led_adoption = 0.02 + 0.005 * i / len(dates)
        
        record = {
            'Date': date,
            'GDP_Growth_Rate': round(gdp_growth, 2),
            'Inflation_Rate': round(inflation_rate, 2),
            'USD_CNY_Exchange_Rate': round(exchange_rate_usd_cny, 3),
            'Consumer_Confidence_Index': round(95 + np.random.uniform(-5, 5), 1),
            'TV_Global_Shipments': tv_shipments_global,
            'Monitor_Demand_Index': round(monitor_demand_index, 1),
            'OLED_Market_Share': round(oled_market_share, 3),
            'MicroLED_Adoption_Rate': round(micro_led_adoption, 3),
            'Display_Technology_Index': round(100 + np.random.uniform(-5, 5), 1),
            'Energy_Cost_Index': round(100 + np.random.uniform(-15, 15), 1)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Usage example and data generation
if __name__ == "__main__":
    print("=== TFT-LCD Synthetic Dataset Generator ===")
    print("Generating comprehensive synthetic dataset...")
    
    # Generate main production dataset
    print("\n1. Generating main production dataset...")
    main_df = create_sample_dataset()
    
    # Generate component dataset
    print("2. Generating component pricing dataset...")
    component_df = create_component_dataset()
    
    # Generate market data
    print("3. Generating market indicators dataset...")
    market_df = create_market_data()
    
    print(f"\n=== Dataset Summary ===")
    print(f"Main Dataset: {main_df.shape[0]} records, {main_df.shape[1]} columns")
    print(f"Component Dataset: {component_df.shape[0]} records, {component_df.shape[1]} columns")
    print(f"Market Dataset: {market_df.shape[0]} records, {market_df.shape[1]} columns")
    
    print(f"\nDate Range: {main_df['Date'].min()} to {main_df['Date'].max()}")
    print(f"Panel Sizes: {sorted(main_df['Panel_Size'].unique())}")
    print(f"Plants: {sorted(main_df['Plant'].unique())}")
    print(f"Market Segments: {sorted(main_df['Market_Segment'].unique())}")
    
    # Show sample data
    print(f"\n=== Sample Main Dataset (first 5 records) ===")
    for i, row in main_df.head().iterrows():
        print(f"Record {i+1}:")
        print(f"  Date: {row['Date'].strftime('%Y-%m-%d')}")
        print(f"  Panel: {row['Panel_Size']}\" {row['Market_Segment']} at {row['Plant']}")
        print(f"  Demand: {row['Forecasted_Demand']:,} | Production: {row['Actual_Production']:,}")
        print(f"  Yield: {row['Production_Yield']:.1%} | Price: ${row['Unit_Selling_Price']:.2f}")
        print(f"  Revenue: ${row['Revenue']:,}")
        print()
    
    # Key statistics
    print(f"=== Key Statistics ===")
    print(f"Average Weekly Demand: {main_df['Forecasted_Demand'].mean():,.0f} units")
    print(f"Average Production Yield: {main_df['Production_Yield'].mean():.1%}")
    print(f"Average Capacity Utilization: {main_df['Capacity_Utilization'].mean():.1%}")
    print(f"Average Unit Price: ${main_df['Unit_Selling_Price'].mean():.2f}")
    print(f"Average Unit Margin: ${main_df['Unit_Margin'].mean():.2f}")
    print(f"Total Revenue (12 weeks): ${main_df['Revenue'].sum():,.0f}")
    
    # Export capability (pseudo-code for actual file saving)
    print(f"\n=== Export Instructions ===")
    print("To save these datasets, use:")
    print("main_df.to_csv('tft_lcd_main_data.csv', index=False)")
    print("component_df.to_csv('tft_lcd_component_data.csv', index=False)")
    print("market_df.to_csv('tft_lcd_market_data.csv', index=False)")
    
    # Return datasets for further use
    # return main_df, component_df, market_df