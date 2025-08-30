import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_tft_lcd_demand_data():
    """
    Generate TFT-LCD demand data with legacy columns structure
    Based on hierarchical optimization repository requirements
    """
    
    # Define parameters
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define categories (matching legacy structure)
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    panel_sizes = ['15.6"', '21.5"', '27"', '32"', '43"', '55"', '65"']
    market_segments = ['Consumer TV', 'Commercial Display', 'Gaming Monitor', 'Professional Monitor', 'Laptop Display']
    product_types = ['Standard', 'Premium', 'Economy']  # Legacy column
    
    # Customer IDs for legacy structure
    customer_ids = [f'CUST_{i:04d}' for i in range(1, 51)]
    
    # Priority levels for legacy structure
    priority_levels = ['High', 'Medium', 'Low']
    
    # Base demand patterns (units per day)
    base_demand = {
        'Consumer TV': {'32"': 800, '43"': 1200, '55"': 1500, '65"': 900},
        'Commercial Display': {'21.5"': 400, '32"': 600, '43"': 700, '55"': 500},
        'Gaming Monitor': {'21.5"': 300, '27"': 800, '32"': 600},
        'Professional Monitor': {'21.5"': 250, '27"': 450, '32"': 350},
        'Laptop Display': {'15.6"': 2000}
    }
    
    # Regional multipliers
    regional_multipliers = {
        'North America': 1.2,
        'Europe': 1.0,
        'Asia Pacific': 1.8,
        'Latin America': 0.6
    }
    
    # Seasonal patterns (monthly multipliers)
    seasonal_patterns = {
        1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.1,
        7: 1.05, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.35, 12: 1.25
    }
    
    # Weekly patterns
    weekly_patterns = {
        0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.15, 5: 1.2, 6: 1.1
    }
    
    # Generate demand data with legacy columns
    data = []
    order_id = 10000  # Starting order ID
    
    for date in date_range:
        month = date.month
        day_of_week = date.dayofweek
        week_num = date.isocalendar()[1]
        
        for region in regions:
            for segment in market_segments:
                if segment in base_demand:
                    for panel_size in base_demand[segment].keys():
                        # Calculate base demand
                        base = base_demand[segment][panel_size]
                        
                        # Apply multipliers
                        regional_mult = regional_multipliers[region]
                        seasonal_mult = seasonal_patterns[month]
                        weekly_mult = weekly_patterns[day_of_week]
                        
                        # Add trend component
                        trend_mult = 1 + (date.timetuple().tm_yday / 365) * 0.05
                        
                        # Calculate forecasted demand
                        forecasted_demand = base * regional_mult * seasonal_mult * weekly_mult * trend_mult
                        forecasted_demand *= np.random.uniform(0.95, 1.05)
                        forecasted_demand = int(forecasted_demand)
                        
                        # Calculate actual demand
                        days_from_start = (date - start_date).days
                        error_std = 0.1 + (days_from_start / 365) * 0.05
                        actual_multiplier = np.random.normal(1.0, error_std)
                        actual_multiplier = max(0.5, min(1.5, actual_multiplier))
                        actual_demand = int(forecasted_demand * actual_multiplier)
                        
                        # Special events
                        if np.random.random() < 0.02:
                            event_multiplier = np.random.choice([0.5, 0.7, 1.3, 1.5])
                            actual_demand = int(actual_demand * event_multiplier)
                        
                        # Legacy columns
                        customer_id = np.random.choice(customer_ids)
                        product_type = np.random.choice(product_types)
                        priority = np.random.choice(priority_levels, p=[0.2, 0.5, 0.3])
                        
                        # Lead time based on priority
                        lead_time_days = {
                            'High': np.random.randint(3, 7),
                            'Medium': np.random.randint(7, 14),
                            'Low': np.random.randint(14, 21)
                        }[priority]
                        
                        # Price calculation (legacy requirement)
                        base_price = {
                            '15.6"': 150, '21.5"': 250, '27"': 350,
                            '32"': 450, '43"': 650, '55"': 850, '65"': 1200
                        }[panel_size]
                        
                        price_multiplier = {
                            'Standard': 1.0, 'Premium': 1.3, 'Economy': 0.8
                        }[product_type]
                        
                        unit_price = base_price * price_multiplier
                        
                        # Inventory levels (legacy columns)
                        safety_stock = int(actual_demand * 0.2)
                        reorder_point = int(actual_demand * 0.3)
                        max_inventory = int(actual_demand * 1.5)
                        
                        # Production time estimate
                        production_hours = actual_demand * 0.1  # 0.1 hours per unit
                        
                        data.append({
                            # Original columns
                            'Date': date.strftime('%Y-%m-%d'),
                            'Region': region,
                            'Panel_Size': panel_size,
                            'Market_Segment': segment,
                            'Forecasted_Demand': forecasted_demand,
                            'Actual_Demand': actual_demand,
                            
                            # Legacy columns
                            'Order_ID': f'ORD_{order_id}',
                            'Customer_ID': customer_id,
                            'Product_Type': product_type,
                            'Priority': priority,
                            'Lead_Time_Days': lead_time_days,
                            'Unit_Price': unit_price,
                            'Total_Value': unit_price * actual_demand,
                            'Week_Number': week_num,
                            'Month': month,
                            'Quarter': (month - 1) // 3 + 1,
                            'Year': date.year,
                            'Safety_Stock': safety_stock,
                            'Reorder_Point': reorder_point,
                            'Max_Inventory': max_inventory,
                            'Production_Hours': production_hours,
                            'Demand_Variance': abs(actual_demand - forecasted_demand),
                            'Forecast_Accuracy': 1 - abs(actual_demand - forecasted_demand) / max(forecasted_demand, 1),
                            'Is_Peak_Season': month in [10, 11, 12],
                            'Day_of_Week': date.strftime('%A'),
                            'Planning_Horizon': 'Short' if lead_time_days < 7 else ('Medium' if lead_time_days < 14 else 'Long')
                        })
                        
                        order_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date and other columns
    df = df.sort_values(['Date', 'Region', 'Market_Segment', 'Panel_Size'])
    
    # Add cumulative columns (legacy requirements)
    df['Cumulative_Demand'] = df.groupby(['Region', 'Panel_Size'])['Actual_Demand'].cumsum()
    df['Cumulative_Revenue'] = df.groupby(['Region', 'Panel_Size'])['Total_Value'].cumsum()
    
    # Save to CSV
    df.to_csv('tft_lcd_demand_data.csv', index=False)
    
    print(f"Generated TFT-LCD demand data with {len(df)} records")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"\nColumns in dataset ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\nData structure preview:")
    print(df.head(5))
    print(f"\nSummary statistics:")
    print(df[['Forecasted_Demand', 'Actual_Demand', 'Unit_Price', 'Total_Value']].describe())
    
    return df

def generate_tft_lcd_plant_data():
    """
    Generate TFT-LCD plant constraint data
    """
    
    plants_data = {
        'Plant_ID': ['PLT_TW01', 'PLT_CN01', 'PLT_CN02', 'PLT_KR01', 'PLT_MX01', 'PLT_VN01'],
        'Plant_Name': ['Taiwan_Fab1', 'China_Fab1', 'China_Fab2', 'Korea_Fab1', 'Mexico_Fab1', 'Vietnam_Fab1'],
        'Location': ['Taiwan', 'Shanghai', 'Shenzhen', 'Seoul', 'Tijuana', 'Ho Chi Minh'],
        'Region': ['Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'Asia Pacific', 'North America', 'Asia Pacific'],
        'Capacity_Per_Day': [5000, 8000, 6000, 4000, 3000, 3500],
        'Max_Panel_Size': ['65"', '65"', '55"', '32"', '32"', '43"'],
        'Min_Panel_Size': ['27"', '21.5"', '32"', '15.6"', '15.6"', '21.5"'],
        'Specialization_1': ['Consumer TV', 'Consumer TV', 'Commercial Display', 'Gaming Monitor', 'Laptop Display', 'Commercial Display'],
        'Specialization_2': ['Gaming Monitor', 'Commercial Display', 'Consumer TV', 'Professional Monitor', 'Commercial Display', 'Professional Monitor'],
        'Production_Cost_Per_Unit': [100, 85, 90, 120, 95, 80],
        'Setup_Cost': [5000, 4000, 4500, 6000, 4500, 3500],
        'Changeover_Time_Hours': [2.0, 1.5, 1.8, 2.5, 2.0, 1.5],
        'Efficiency': [0.95, 0.92, 0.93, 0.98, 0.90, 0.88],
        'Energy_Cost_Per_Unit': [15, 12, 13, 18, 14, 11],
        'Labor_Cost_Per_Hour': [45, 25, 28, 50, 35, 20],
        'Max_Workforce': [500, 800, 650, 400, 350, 400],
        'Current_Workforce': [450, 750, 600, 380, 320, 360],
        'Maintenance_Cost_Daily': [2000, 2500, 2200, 1800, 1500, 1600],
        'Quality_Rate': [0.98, 0.96, 0.97, 0.99, 0.95, 0.94],
        'Lead_Time_Days': [2, 3, 3, 2, 4, 5],
        'Min_Batch_Size': [100, 150, 120, 80, 100, 110],
        'Max_Batch_Size': [2000, 3000, 2500, 1500, 1200, 1400],
        'Storage_Capacity': [15000, 25000, 20000, 12000, 10000, 12000],
        'Operating_Days_Per_Year': [350, 340, 345, 355, 330, 335],
        'Shifts_Per_Day': [3, 3, 3, 2, 2, 3],
        'Hours_Per_Shift': [8, 8, 8, 12, 12, 8],
        'Technology_Generation': ['Gen 10.5', 'Gen 8.5', 'Gen 10.5', 'Gen 8.5', 'Gen 6', 'Gen 7'],
        'Automation_Level': ['High', 'Medium', 'High', 'High', 'Medium', 'Low'],
        'ISO_Certified': [True, True, True, True, True, False],
        'Carbon_Emission_Per_Unit': [2.5, 3.2, 2.8, 2.2, 3.5, 4.0],
        'Water_Usage_Per_Unit': [50, 65, 55, 45, 70, 80],
        'Flexibility_Score': [0.85, 0.75, 0.80, 0.90, 0.70, 0.65],
        'Reliability_Score': [0.95, 0.90, 0.92, 0.98, 0.88, 0.85],
        'Distance_To_Port_KM': [50, 100, 80, 30, 20, 120],
        'Year_Established': [2015, 2010, 2018, 2012, 2016, 2020],
        'Last_Upgrade_Year': [2022, 2021, 2023, 2022, 2020, 2021],
        'Expansion_Potential': ['Medium', 'High', 'Medium', 'Low', 'High', 'High']
    }
    
    # Create DataFrame
    df = pd.DataFrame(plants_data)
    
    # Add calculated columns
    df['Daily_Operating_Hours'] = df['Shifts_Per_Day'] * df['Hours_Per_Shift']
    df['Annual_Capacity'] = df['Capacity_Per_Day'] * df['Operating_Days_Per_Year']
    df['Effective_Capacity'] = df['Annual_Capacity'] * df['Efficiency']
    df['Total_Cost_Per_Unit'] = df['Production_Cost_Per_Unit'] + df['Energy_Cost_Per_Unit']
    df['Workforce_Utilization'] = df['Current_Workforce'] / df['Max_Workforce']
    df['Age_Years'] = 2024 - df['Year_Established']
    df['Years_Since_Upgrade'] = 2024 - df['Last_Upgrade_Year']
    
    # Add panel size capabilities (binary flags)
    panel_sizes = ['15.6"', '21.5"', '27"', '32"', '43"', '55"', '65"']
    size_map = {'15.6"': 1, '21.5"': 2, '27"': 3, '32"': 4, '43"': 5, '55"': 6, '65"': 7}
    
    for size in panel_sizes:
        col_name = f'Can_Produce_{size.replace(".", "_").replace('"', "in")}'
        df[col_name] = df.apply(
            lambda row: size_map[size] >= size_map.get(row['Min_Panel_Size'], 8) and 
                       size_map[size] <= size_map.get(row['Max_Panel_Size'], 0),
            axis=1
        )
    
    # Save to CSV
    df.to_csv('tft_lcd_plant_data.csv', index=False)
    
    print(f"\nGenerated TFT-LCD plant data with {len(df)} plants")
    print(f"Columns in plant dataset ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\nPlant summary:")
    print(df[['Plant_Name', 'Location', 'Capacity_Per_Day', 'Efficiency', 'Quality_Rate']].to_string())
    
    return df

if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING TFT-LCD DATA FILES")
    print("=" * 80)
    
    # Generate demand data
    print("\n1. Generating Demand Data...")
    demand_df = generate_tft_lcd_demand_data()
    
    # Generate plant data
    print("\n2. Generating Plant Constraint Data...")
    plant_df = generate_tft_lcd_plant_data()
    
    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETE")
    print("=" * 80)
    print("\nFiles created:")
    print("  1. tft_lcd_demand_data.csv - Demand data with legacy columns")
    print("  2. tft_lcd_plant_data.csv - Plant constraints and capabilities")
    print("\nYou can now use these files with the hierarchical optimizer.")