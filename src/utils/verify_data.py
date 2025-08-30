import pandas as pd
import numpy as np

# Set seed for reproducibility
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
                
                # Create record
                record = {
                    'Date': date,
                    'Week': week,
                    'Panel_Size': size,
                    'Plant': plant,
                    'Market_Segment': segment,
                    'Forecasted_Demand': round(demand),
                    'Actual_Production': round(actual_production),
                    'Production_Yield': round(yield_rate, 3),
                    'Good_Units_Produced': round(good_units),
                    'Actual_Sales': round(sales),
                    'Unit_Selling_Price': round(unit_price, 2),
                    'Unit_Production_Cost': round(production_cost, 2),
                    'Revenue': round(sales * unit_price),
                    'Supply_Disruptions': disruption,
                    'On_Time_Delivery': round(on_time_delivery, 3)
                }
                
                data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

print("TFT-LCD Synthetic Dataset Generated Successfully!")
print(f"Dataset Shape: {df.shape}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total Records: {len(df):,}")

print("\n=== Dataset Overview ===")
print(f"Panel Sizes: {sorted(df['Panel_Size'].unique())}")
print(f"Plants: {list(df['Plant'].unique())}")
print(f"Market Segments: {list(df['Market_Segment'].unique())}")

print("\n=== Sample Data (First 8 Records) ===")
df.head(8)