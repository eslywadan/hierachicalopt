"""
H2Opt with Demand Smoothing: Level 1 adjusts due dates to balance capacity
This implements realistic demand management strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from ortools.sat.python import cp_model

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class DemandSmoother:
    """Smooths demand peaks by adjusting due dates within lead time windows"""
    
    @staticmethod
    def analyze_demand_profile(demand_df, plant_df, window_days=7):
        """Analyze demand profile and identify peaks"""
        print("\n" + "="*80)
        print("DEMAND PROFILE ANALYSIS")
        print("="*80)
        
        # Calculate daily capacity
        total_daily_capacity = sum(
            plant['Capacity_Per_Day'] * plant['Efficiency'] 
            for _, plant in plant_df.iterrows()
        )
        
        # Group demand by date
        daily_demand = demand_df.groupby('Date')['Forecasted_Demand'].sum().reset_index()
        daily_demand['Date'] = pd.to_datetime(daily_demand['Date'])
        
        # Calculate rolling average
        daily_demand['Rolling_Avg'] = daily_demand['Forecasted_Demand'].rolling(window=window_days, min_periods=1).mean()
        
        # Identify peaks and valleys
        daily_demand['Capacity_Ratio'] = daily_demand['Forecasted_Demand'] / total_daily_capacity
        
        print(f"\nDaily Capacity: {total_daily_capacity:,.0f} units")
        print(f"Average Daily Demand: {daily_demand['Forecasted_Demand'].mean():,.0f} units")
        print(f"Peak Daily Demand: {daily_demand['Forecasted_Demand'].max():,.0f} units")
        print(f"Min Daily Demand: {daily_demand['Forecasted_Demand'].min():,.0f} units")
        
        # Find overload days
        overload_days = daily_demand[daily_demand['Capacity_Ratio'] > 1.0]
        print(f"\nDays with demand > capacity: {len(overload_days)} out of {len(daily_demand)}")
        
        if len(overload_days) > 0:
            print(f"Average overload: {(overload_days['Capacity_Ratio'].mean() - 1) * 100:.1f}% above capacity")
        
        return daily_demand, total_daily_capacity
    
    @staticmethod
    def smooth_demand_by_priority(demand_df, plant_df, max_shift_days=7):
        """
        Smooth demand by shifting low-priority orders within lead time windows
        """
        print("\n" + "="*80)
        print("DEMAND SMOOTHING BY PRIORITY")
        print("="*80)
        
        demand_smooth = demand_df.copy()
        demand_smooth['Date'] = pd.to_datetime(demand_smooth['Date'])
        demand_smooth['Original_Date'] = demand_smooth['Date']
        
        # Calculate daily capacity
        total_daily_capacity = sum(
            plant['Capacity_Per_Day'] * plant['Efficiency'] 
            for _, plant in plant_df.iterrows()
        )
        
        # Sort by date and priority (High priority first)
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        demand_smooth['Priority_Rank'] = demand_smooth['Priority'].map(priority_order)
        demand_smooth = demand_smooth.sort_values(['Date', 'Priority_Rank'])
        
        # Track daily allocations
        daily_allocated = {}
        shifts_made = 0
        
        for idx, row in demand_smooth.iterrows():
            current_date = row['Date']
            demand_amount = row['Forecasted_Demand']
            priority = row['Priority']
            lead_time = row['Lead_Time_Days']
            
            # Check if current date has capacity
            date_str = current_date.strftime('%Y-%m-%d')
            if date_str not in daily_allocated:
                daily_allocated[date_str] = 0
            
            remaining_capacity = total_daily_capacity - daily_allocated[date_str]
            
            if demand_amount <= remaining_capacity:
                # Can fulfill on requested date
                daily_allocated[date_str] += demand_amount
            else:
                # Need to find alternative date
                if priority == 'Low':
                    # Try to shift low priority orders
                    max_shift = min(max_shift_days, lead_time // 2)  # Can shift up to half of lead time
                    
                    # Look for available capacity in next few days
                    found_slot = False
                    for shift in range(1, max_shift + 1):
                        new_date = current_date + timedelta(days=shift)
                        new_date_str = new_date.strftime('%Y-%m-%d')
                        
                        if new_date_str not in daily_allocated:
                            daily_allocated[new_date_str] = 0
                        
                        if daily_allocated[new_date_str] + demand_amount <= total_daily_capacity:
                            # Found available slot
                            demand_smooth.at[idx, 'Date'] = new_date
                            daily_allocated[new_date_str] += demand_amount
                            shifts_made += 1
                            found_slot = True
                            break
                    
                    if not found_slot:
                        # No slot found, keep original date but will exceed capacity
                        daily_allocated[date_str] += demand_amount
                
                elif priority == 'Medium' and remaining_capacity < demand_amount * 0.5:
                    # For medium priority, only shift if really necessary
                    max_shift = min(max_shift_days // 2, lead_time // 3)
                    
                    for shift in range(1, max_shift + 1):
                        new_date = current_date + timedelta(days=shift)
                        new_date_str = new_date.strftime('%Y-%m-%d')
                        
                        if new_date_str not in daily_allocated:
                            daily_allocated[new_date_str] = 0
                        
                        if daily_allocated[new_date_str] + demand_amount <= total_daily_capacity * 0.9:
                            demand_smooth.at[idx, 'Date'] = new_date
                            daily_allocated[new_date_str] += demand_amount
                            shifts_made += 1
                            break
                    else:
                        daily_allocated[date_str] += demand_amount
                else:
                    # High priority - never shift
                    daily_allocated[date_str] += demand_amount
        
        print(f"\nSmoothing Results:")
        print(f"  Orders shifted: {shifts_made}")
        print(f"  Percentage shifted: {(shifts_made / len(demand_smooth)) * 100:.1f}%")
        
        # Calculate improvement
        original_overload = sum(
            max(0, alloc - total_daily_capacity) 
            for alloc in daily_allocated.values()
        )
        
        print(f"  Total overload after smoothing: {original_overload:,.0f} units")
        
        # Add shift information
        demand_smooth['Days_Shifted'] = (demand_smooth['Date'] - demand_smooth['Original_Date']).dt.days
        
        return demand_smooth
    
    @staticmethod
    def create_balanced_weekly_demand(demand_df, plant_df, planning_days=7):
        """
        Create balanced weekly demand windows for optimization
        """
        print("\n" + "="*80)
        print("CREATING BALANCED WEEKLY WINDOWS")
        print("="*80)
        
        # First smooth the demand
        demand_smooth = DemandSmoother.smooth_demand_by_priority(demand_df, plant_df)
        
        # Calculate weekly capacity
        total_weekly_capacity = sum(
            plant['Capacity_Per_Day'] * plant['Efficiency'] * planning_days
            for _, plant in plant_df.iterrows()
        )
        
        # Group into weekly windows
        demand_smooth['Week'] = pd.to_datetime(demand_smooth['Date']).dt.isocalendar().week
        weekly_demand = demand_smooth.groupby('Week')['Forecasted_Demand'].sum().reset_index()
        
        print(f"\nWeekly Statistics:")
        print(f"  Weekly Capacity: {total_weekly_capacity:,.0f} units")
        print(f"  Average Weekly Demand: {weekly_demand['Forecasted_Demand'].mean():,.0f} units")
        print(f"  Peak Weekly Demand: {weekly_demand['Forecasted_Demand'].max():,.0f} units")
        
        # Identify weeks that can be optimized
        feasible_weeks = weekly_demand[weekly_demand['Forecasted_Demand'] <= total_weekly_capacity * 1.2]
        print(f"  Feasible weeks (demand < 120% capacity): {len(feasible_weeks)} out of {len(weekly_demand)}")
        
        return demand_smooth, weekly_demand


class Level1StrategicWithSmoothing(Problem):
    """Level 1 optimization with integrated demand smoothing"""
    
    def __init__(self, demand_df, plant_df, planning_horizon=30, enable_smoothing=True):
        self.demand_df_original = demand_df
        self.plant_df = plant_df
        self.planning_horizon = planning_horizon
        self.enable_smoothing = enable_smoothing
        
        # Apply demand smoothing if enabled
        if enable_smoothing:
            self.demand_df = DemandSmoother.smooth_demand_by_priority(
                demand_df, plant_df, max_shift_days=7
            )
        else:
            self.demand_df = demand_df
        
        # Aggregate demand
        self.agg_demand = self._aggregate_demand()
        self.total_demand = self.agg_demand['Forecasted_Demand'].sum()
        
        # Calculate capacity
        self.total_capacity = sum(
            row['Capacity_Per_Day'] * planning_horizon * row['Efficiency']
            for _, row in plant_df.iterrows()
        )
        
        print(f"\n  Level 1 Setup:")
        print(f"    Total Demand (30 days): {self.total_demand:,.0f} units")
        print(f"    Total Capacity (30 days): {self.total_capacity:,.0f} units")
        print(f"    Required Utilization: {(self.total_demand/self.total_capacity)*100:.1f}%")
        
        n_vars = len(plant_df)
        n_obj = 3
        n_constr = 3
        
        # Ensure minimum allocation to each plant
        xl = np.ones(n_vars) * 0.05  # At least 5% to each plant
        xu = np.ones(n_vars)
        
        super().__init__(n_var=n_vars, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
    
    def _aggregate_demand(self):
        """Aggregate smoothed demand"""
        start_date = pd.to_datetime(self.demand_df['Date'].min())
        end_date = start_date + timedelta(days=self.planning_horizon)
        
        period_demand = self.demand_df[
            (pd.to_datetime(self.demand_df['Date']) >= start_date) & 
            (pd.to_datetime(self.demand_df['Date']) < end_date)
        ]
        
        return period_demand.groupby(['Market_Segment', 'Panel_Size']).agg({
            'Forecasted_Demand': 'sum',
            'Total_Value': 'sum'
        }).reset_index()
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate with capacity-aware objectives"""
        pop_size = x.shape[0]
        
        f1_cost = np.zeros(pop_size)
        f2_utilization = np.zeros(pop_size)
        f3_unmet = np.zeros(pop_size)
        
        g1_capacity = np.zeros(pop_size)
        g2_min_util = np.zeros(pop_size)
        g3_balance = np.zeros(pop_size)
        
        for i in range(pop_size):
            allocation = x[i]
            
            total_cost = 0
            total_capacity_used = 0
            total_producible = 0
            
            for j, plant_row in self.plant_df.iterrows():
                plant_capacity = plant_row['Capacity_Per_Day'] * self.planning_horizon * plant_row['Efficiency']
                allocated_capacity = allocation[j] * plant_capacity
                
                # Cost calculation
                production_cost = allocated_capacity * plant_row['Total_Cost_Per_Unit']
                setup_cost = plant_row['Setup_Cost'] * self.planning_horizon * allocation[j]
                maintenance_cost = plant_row['Maintenance_Cost_Daily'] * self.planning_horizon * allocation[j]
                
                total_cost += production_cost + setup_cost + maintenance_cost
                total_capacity_used += allocated_capacity
                
                # Estimate production based on capabilities
                for _, demand_row in self.agg_demand.iterrows():
                    panel_size = demand_row['Panel_Size']
                    col_name = f"Can_Produce_{panel_size.replace('.', '_').replace('\"', 'in')}"
                    
                    if col_name in plant_row and plant_row[col_name]:
                        product_share = demand_row['Forecasted_Demand'] / self.total_demand
                        total_producible += min(
                            allocated_capacity * product_share,
                            demand_row['Forecasted_Demand'] * allocation[j]
                        )
            
            # Objectives
            f1_cost[i] = total_cost
            f2_utilization[i] = -total_capacity_used / self.total_capacity  # Maximize utilization
            f3_unmet[i] = max(0, self.total_demand - total_producible)
            
            # Constraints
            g1_capacity[i] = total_capacity_used - self.total_capacity  # <= total capacity
            
            # Ensure sufficient utilization if demand requires it
            min_required_util = min(0.95, self.total_demand / self.total_capacity)
            g2_min_util[i] = min_required_util - (total_capacity_used / self.total_capacity)
            
            # Balance constraint - standard deviation should be reasonable
            g3_balance[i] = np.std(allocation) - 0.3  # Std dev <= 0.3
        
        out["F"] = np.column_stack([f1_cost, f2_utilization, f3_unmet])
        out["G"] = np.column_stack([g1_capacity, g2_min_util, g3_balance])


class H2Opt:

    """H2Opt with integrated demand smoothing capabilities"""
    
    def __init__(self, demand_file='tft_lcd_demand_data.csv', plant_file='tft_lcd_plant_data.csv', enable_smoothing=True):
        self.plant_df = pd.read_csv(plant_file)
        self.demand_df_original = pd.read_csv(demand_file)
        self.demand_df_original['Date'] = pd.to_datetime(self.demand_df_original['Date'])
        
        # Analyze demand profile
        if enable_smoothing:
            print("\nDemand smoothing is ENABLED.")
            self.daily_demand, self.daily_capacity = DemandSmoother.analyze_demand_profile(
                self.demand_df_original, self.plant_df
            )
        
            # Create smoothed demand
            self.demand_df_smoothed, self.weekly_demand = DemandSmoother.create_balanced_weekly_demand(
                self.demand_df_original, self.plant_df
            )
        
        # Use smoothed demand for optimization
        self.demand_df = self.demand_df_smoothed
        
        self.surrogate_model = None
        self.scaler = StandardScaler()
    
    def level1_strategic_optimization(self, planning_horizon=30):
        """Level 1 with demand smoothing"""
        print("\n" + "="*80)
        print("LEVEL 1: STRATEGIC PLANNING WITH DEMAND SMOOTHING")
        print("="*80)
        
        # Use first month of smoothed demand
        problem = Level1StrategicWithSmoothing(
            self.demand_df, self.plant_df, planning_horizon, enable_smoothing=False
        )
        
        algo = NSGA2(
            pop_size=150,  # Larger population for better exploration
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        print(f"\nRunning strategic optimization with smoothed demand...")
        
        res = minimize(
            problem,
            algo,
            ('n_gen', 100),
            seed=42,
            verbose=False
        )
        
        capacity_allocation = {}
        
        if res.F is not None and len(res.F) > 0:
            # Select solution with best balance of objectives
            # Prioritize meeting demand over pure cost
            costs = res.F[:, 0]
            utilizations = -res.F[:, 1]
            unmet_demands = res.F[:, 2]
            
            # Normalize objectives
            costs_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
            util_norm = (utilizations - utilizations.min()) / (utilizations.max() - utilizations.min() + 1e-6)
            unmet_norm = (unmet_demands - unmet_demands.min()) / (unmet_demands.max() - unmet_demands.min() + 1e-6)
            
            # Weight: prioritize meeting demand (low unmet), then utilization, then cost
            scores = 0.2 * costs_norm - 0.3 * util_norm + 0.5 * unmet_norm
            best_idx = scores.argmin()
            
            best_solution = res.X[best_idx]
            best_objectives = res.F[best_idx]
            
            print(f"\nOptimization Results:")
            print(f"  Total Cost: ${best_objectives[0]:,.0f}")
            print(f"  Capacity Utilization: {-best_objectives[1]*100:.1f}%")
            print(f"  Unmet Demand: {best_objectives[2]:,.0f} units")
            
            print(f"\nPlant Allocations:")
            total_allocated = 0
            
            for i, plant_row in self.plant_df.iterrows():
                plant_capacity = plant_row['Capacity_Per_Day'] * planning_horizon * plant_row['Efficiency']
                allocated = best_solution[i] * plant_capacity
                total_allocated += allocated
                
                capacity_allocation[plant_row['Plant_Name']] = {
                    'allocation_ratio': best_solution[i],
                    'allocated_capacity': allocated,
                    'max_capacity': plant_capacity,
                    'utilization': (allocated / plant_capacity) * 100
                }
                
                status = "✓" if best_solution[i] >= 0.05 else "⚠"
                print(f"  {status} {plant_row['Plant_Name']:15s}: {allocated:8,.0f} units ({best_solution[i]*100:5.1f}% allocation)")
            
            print(f"\nTotal Allocated: {total_allocated:,.0f} units")
            print(f"Demand Coverage: {(total_allocated/problem.total_demand)*100:.1f}%")
        
        return capacity_allocation, problem.agg_demand
    
    def level2_tactical_optimization_adaptive(self, capacity_allocation, agg_demand, planning_days=7):
        """Adaptive Level 2 that adjusts to available capacity"""
        print("\n" + "="*80)
        print("LEVEL 2: ADAPTIVE TACTICAL ALLOCATION")
        print("="*80)
        
        # Calculate actual capacity vs demand
        total_weekly_demand = agg_demand['Forecasted_Demand'].sum() / 30 * planning_days
        total_weekly_capacity = sum(
            cap['allocated_capacity'] / 30 * planning_days 
            for cap in capacity_allocation.values()
        )
        
        coverage_ratio = total_weekly_capacity / total_weekly_demand
        
        print(f"\nCapacity Analysis:")
        print(f"  Weekly demand: {total_weekly_demand:,.0f} units")
        print(f"  Weekly capacity: {total_weekly_capacity:,.0f} units")
        print(f"  Coverage ratio: {coverage_ratio:.2%}")
        
        if coverage_ratio < 0.5:
            print(f"\n⚠ Low coverage. Implementing demand prioritization...")
            return self._prioritized_allocation(capacity_allocation, agg_demand, planning_days)
        else:
            print(f"\n✓ Sufficient coverage. Running CP optimization...")
            return self._cp_optimization(capacity_allocation, agg_demand, planning_days, coverage_ratio)
    
    def _prioritized_allocation(self, capacity_allocation, agg_demand, planning_days):
        """Allocate based on priority when capacity is very limited"""
        allocations = []
        
        # Sort demand by value/priority
        agg_demand_sorted = agg_demand.sort_values('Total_Value', ascending=False)
        
        # Track remaining capacity per plant
        remaining_capacity = {}
        for plant_name, cap_info in capacity_allocation.items():
            remaining_capacity[plant_name] = cap_info['allocated_capacity'] / 30 * planning_days
        
        for _, demand_row in agg_demand_sorted.iterrows():
            weekly_demand = demand_row['Forecasted_Demand'] / 30 * planning_days
            panel_size = demand_row['Panel_Size']
            
            plant_allocations = {}
            demand_remaining = weekly_demand
            
            # Find capable plants with capacity
            for plant_name, capacity_left in remaining_capacity.items():
                if capacity_left <= 0 or demand_remaining <= 0:
                    continue
                
                plant_row = self.plant_df[self.plant_df['Plant_Name'] == plant_name].iloc[0]
                col_name = f"Can_Produce_{panel_size.replace('.', '_').replace('\"', 'in')}"
                
                if col_name in plant_row and plant_row[col_name]:
                    # Allocate what we can
                    allocated = min(demand_remaining, capacity_left * 0.2)  # Max 20% per product
                    if allocated > 10:
                        plant_allocations[plant_name] = allocated
                        remaining_capacity[plant_name] -= allocated
                        demand_remaining -= allocated
            
            allocations.append({
                'Segment': demand_row['Market_Segment'],
                'Panel_Size': panel_size,
                'Total_Demand': weekly_demand,
                'Allocations': plant_allocations,
                'Unmet_Demand': max(0, demand_remaining),
                'Fill_Rate': ((weekly_demand - demand_remaining) / weekly_demand * 100) if weekly_demand > 0 else 0
            })
        
        return allocations
    
    def _cp_optimization(self, capacity_allocation, agg_demand, planning_days, coverage_ratio):
        """Run CP optimization with appropriate constraints"""
        model = cp_model.CpModel()
        
        plants = list(capacity_allocation.keys())
        products = list(range(len(agg_demand)))
        
        # Decision variables
        allocation = {}
        for p in products:
            allocation[p] = {}
            for i, plant_name in enumerate(plants):
                panel_size = agg_demand.iloc[p]['Panel_Size']
                plant_row = self.plant_df[self.plant_df['Plant_Name'] == plant_name].iloc[0]
                
                col_name = f"Can_Produce_{panel_size.replace('.', '_').replace('\"', 'in')}"
                can_produce = plant_row[col_name] if col_name in plant_row.index else False
                
                if can_produce:
                    max_alloc = int(capacity_allocation[plant_name]['allocated_capacity'] / 30 * planning_days)
                    allocation[p][i] = model.NewIntVar(0, max_alloc, f'alloc_{p}_{i}')
                else:
                    allocation[p][i] = model.NewIntVar(0, 0, f'alloc_{p}_{i}')
        
        # Demand constraints with slack
        slack_vars = []
        target_fill_rate = min(0.9, coverage_ratio)  # Realistic target based on capacity
        
        for p in products:
            weekly_demand = int(agg_demand.iloc[p]['Forecasted_Demand'] / 30 * planning_days)
            min_satisfy = int(weekly_demand * target_fill_rate)
            
            slack = model.NewIntVar(0, weekly_demand, f'slack_{p}')
            slack_vars.append(slack)
            
            model.Add(sum(allocation[p][i] for i in range(len(plants))) + slack >= min_satisfy)
        
        # Capacity constraints
        for i, plant_name in enumerate(plants):
            weekly_capacity = int(capacity_allocation[plant_name]['allocated_capacity'] / 30 * planning_days)
            model.Add(sum(allocation[p][i] for p in products) <= weekly_capacity)
        
        # Objective
        total_cost = []
        for p in products:
            for i, plant_name in enumerate(plants):
                plant_row = self.plant_df[self.plant_df['Plant_Name'] == plant_name].iloc[0]
                cost = int(plant_row['Total_Cost_Per_Unit'])
                total_cost.append(allocation[p][i] * cost)
        
        # Penalty for slack
        for slack in slack_vars:
            total_cost.append(slack * 500)
        
        model.Minimize(sum(total_cost))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(model)
        
        allocations = []
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"✓ CP solution found!")
            
            for p in products:
                demand_row = agg_demand.iloc[p]
                plant_allocations = {}
                
                for i, plant_name in enumerate(plants):
                    allocated = solver.Value(allocation[p][i]) if (p, i) in [(p, i) for p in products for i in range(len(plants)) if (p, i) in allocation] else 0
                    if allocated > 10:
                        plant_allocations[plant_name] = allocated
                
                weekly_demand = demand_row['Forecasted_Demand'] / 30 * planning_days
                total_allocated = sum(plant_allocations.values())
                
                allocations.append({
                    'Segment': demand_row['Market_Segment'],
                    'Panel_Size': demand_row['Panel_Size'],
                    'Total_Demand': weekly_demand,
                    'Allocations': plant_allocations,
                    'Unmet_Demand': max(0, weekly_demand - total_allocated),
                    'Fill_Rate': (total_allocated / weekly_demand * 100) if weekly_demand > 0 else 0
                })
        else:
            print(f"⚠ No CP solution. Using prioritized allocation.")
            allocations = self._prioritized_allocation(capacity_allocation, agg_demand, planning_days)
        
        return allocations
    
    def level3_operational_scheduling(self, allocations):
        """Simple scheduling for allocated products"""
        print("\n" + "="*80)
        print("LEVEL 3: OPERATIONAL SCHEDULING")
        print("="*80)
        
        schedules = {}
        
        for alloc in allocations:
            for plant_name, quantity in alloc['Allocations'].items():
                if plant_name not in schedules:
                    schedules[plant_name] = []
                
                schedules[plant_name].append({
                    'product': f"{alloc['Segment']}_{alloc['Panel_Size']}",
                    'quantity': quantity,
                    'fill_rate': alloc['Fill_Rate']
                })
        
        print(f"\nScheduled {len(schedules)} plants:")
        for plant, products in schedules.items():
            total = sum(p['quantity'] for p in products)
            print(f"  {plant}: {len(products)} products, {total:,.0f} units")
        
        return schedules
    
    def run_optimization(self, level2_opt='adaptive'):
        """Run complete optimization with demand smoothing"""
        print("\n" + "="*100)
        print("H2OPT WITH DEMAND SMOOTHING")
        print("="*100)
        
        # Level 1 - Strategic with smoothed demand
        capacity_allocation, agg_demand = self.level1_strategic_optimization(planning_horizon=30)
        
        # Level 2 - Adaptive tactical
        if level2_opt == 'adaptive':
            allocations = self.level2_tactical_optimization_adaptive(
                capacity_allocation, agg_demand, planning_days=7
            )

        if level2_opt == 'realistic':
            allocations = level2_realistic_economics_optimization(
                capacity_allocation=capacity_allocation, 
                agg_demand=agg_demand, 
                plant_df=self.plant_df, planning_days=7)
        
        # Level 3 - Operational
        schedules = self.level3_operational_scheduling(allocations)
        
        # Calculate metrics
        total_demand = sum(a['Total_Demand'] for a in allocations)
        total_allocated = sum(sum(a['Allocations'].values()) for a in allocations)
        fill_rate = (total_allocated / total_demand * 100) if total_demand > 0 else 0
        
        print("\n" + "="*100)
        print("OPTIMIZATION COMPLETE")
        print("="*100)
        
        print(f"\nFinal Metrics:")
        print(f"  Overall Fill Rate: {fill_rate:.1f}%")
        print(f"  Active Plants: {len(schedules)}")
        print(f"  Demand Smoothing Applied: Yes")
        
        # Show smoothing impact
        shifts_made = len(self.demand_df[self.demand_df['Days_Shifted'] > 0])
        print(f"  Orders shifted: {shifts_made} ({shifts_made/len(self.demand_df)*100:.1f}%)")
        
        return {
            'strategic': capacity_allocation,
            'tactical': allocations,
            'operational': schedules,
            'metrics': {
                'fill_rate': fill_rate,
                'active_plants': len(schedules),
                'orders_shifted': shifts_made
            }
        }

class RealisticEconomicsModel:
    """
    Calculates realistic plant economics including fixed and variable costs
    Maintains consistency with the original class structure
    """
    
    @staticmethod
    def calculate_fixed_costs(plant_df, planning_days=7):
        """
        Calculate fixed costs that occur regardless of production
        Returns INTEGER values for all costs
        """
        fixed_costs = {}
        
        print("\n" + "="*80)
        print("FIXED COST CALCULATION (Per Week - Integer Values)")
        print("="*80)
        
        for _, plant in plant_df.iterrows():
            plant_name = plant['Plant_Name']
            
            # Asset depreciation - INTEGER
            equipment_value = int(plant['Annual_Capacity'] * 500)
            daily_depreciation = equipment_value / (10 * 365)
            weekly_depreciation = int(daily_depreciation * planning_days)
            
            # Fixed labor costs - INTEGER
            management_staff = int(plant['Max_Workforce'] * 0.3)
            avg_salary_per_week = plant['Labor_Cost_Per_Hour'] * 40 * 2
            weekly_salary_cost = int(management_staff * avg_salary_per_week)
            
            # Facility costs - INTEGER
            facility_size_factor = plant['Storage_Capacity'] / 10000
            weekly_facility_cost = int(50000 * facility_size_factor)
            
            # Maintenance base cost - INTEGER
            weekly_maintenance_base = int(plant['Maintenance_Cost_Daily'] * planning_days * 0.5)
            
            # Technology licensing costs - INTEGER
            if plant['Technology_Generation'] in ['Gen 10.5', 'Gen 8.5']:
                weekly_tech_cost = 100000
            else:
                weekly_tech_cost = 30000
            
            # Total fixed costs - INTEGER
            total_fixed = int(weekly_depreciation + weekly_salary_cost + 
                            weekly_facility_cost + weekly_maintenance_base + weekly_tech_cost)
            
            fixed_costs[plant_name] = {
                'depreciation': weekly_depreciation,
                'salaries': weekly_salary_cost,
                'facility': weekly_facility_cost,
                'maintenance_base': weekly_maintenance_base,
                'technology': weekly_tech_cost,
                'total': total_fixed,
                'daily_fixed': total_fixed // planning_days  # Integer division
            }
            
            print(f"\n{plant_name}:")
            print(f"  Depreciation:     ${weekly_depreciation:,d}")
            print(f"  Fixed Salaries:   ${weekly_salary_cost:,d}")
            print(f"  Facility Costs:   ${weekly_facility_cost:,d}")
            print(f"  Base Maintenance: ${weekly_maintenance_base:,d}")
            print(f"  Technology:       ${weekly_tech_cost:,d}")
            print(f"  TOTAL FIXED:      ${total_fixed:,d}/week")
            print(f"  Break-even units: {total_fixed//int(plant['Total_Cost_Per_Unit']):,d}")
        
        return fixed_costs
    
    @staticmethod
    def calculate_production_value(agg_demand, panel_market_values=None):
        """
        Calculate the market value of production
        Returns INTEGER values for all prices
        """
        if panel_market_values is None:
            # Default market values by panel size - INTEGERS
            panel_market_values = {
                '15.6"': 200,
                '21.5"': 350,
                '27"': 500,
                '32"': 650,
                '43"': 900,
                '55"': 1200,
                '65"': 1800
            }
        
        print("\n" + "="*80)
        print("PRODUCTION VALUE CALCULATION (Integer Prices)")
        print("="*80)
        
        product_values = {}
        
        # Segment multipliers
        segment_multipliers = {
            'Consumer TV': 1.0,
            'Gaming Monitor': 1.3,
            'Professional Monitor': 1.5,
            'Commercial Display': 1.2,
            'Laptop Display': 0.9
        }
        
        for idx, (_, row) in enumerate(agg_demand.iterrows()):
            key = f"{row['Market_Segment']}_{row['Panel_Size']}"
            
            # Base value from panel size
            base_value = panel_market_values.get(row['Panel_Size'], 500)
            
            # Segment multiplier
            segment_mult = segment_multipliers.get(row['Market_Segment'], 1.0)
            
            # Calculate total value as INTEGER
            unit_value = int(base_value * segment_mult)
            product_values[key] = unit_value
            
            if idx < 5:  # Show first 5
                print(f"  {key}: ${unit_value:,d}/unit")
        
        return product_values
    
    @staticmethod
    def validate_integer_allocations(allocations):
        """
        Validate that all allocations contain only integer values
        """
        validation_passed = True
        issues = []
        
        for idx, alloc in enumerate(allocations):
            # Check Total_Demand
            if not isinstance(alloc['Total_Demand'], (int, np.integer)):
                issues.append(f"Row {idx}: Total_Demand is {type(alloc['Total_Demand']).__name__} = {alloc['Total_Demand']}")
                validation_passed = False
            
            # Check Unmet_Demand
            if not isinstance(alloc['Unmet_Demand'], (int, np.integer)):
                issues.append(f"Row {idx}: Unmet_Demand is {type(alloc['Unmet_Demand']).__name__} = {alloc['Unmet_Demand']}")
                validation_passed = False
            
            # Check each allocation quantity
            for plant, qty in alloc['Allocations'].items():
                if not isinstance(qty, (int, np.integer)):
                    issues.append(f"Row {idx}: Allocation[{plant}] is {type(qty).__name__} = {qty}")
                    validation_passed = False
                elif qty < 1:
                    issues.append(f"Row {idx}: Allocation[{plant}] has invalid quantity = {qty}")
                    validation_passed = False
        
        return validation_passed, issues


def level2_realistic_economics_optimization(capacity_allocation, agg_demand, plant_df, planning_days=7):
    """
    Level 2 optimization with realistic economics and INTEGER constraints
    Uses the RealisticEconomicsModel class for consistency
    """
    print("\n" + "="*80)
    print("LEVEL 2: REALISTIC ECONOMIC OPTIMIZATION (INTEGER)")
    print("="*80)
    
    # Use class methods to calculate economics
    fixed_costs = RealisticEconomicsModel.calculate_fixed_costs(plant_df, planning_days)
    product_values = RealisticEconomicsModel.calculate_production_value(agg_demand)
    
    # Prepare data with INTEGER conversion
    plants = list(capacity_allocation.keys())
    n_plants = len(plants)
    
    # Create product list with INTEGER demand
    products = []
    product_info = []
    
    for idx, (_, row) in enumerate(agg_demand.iterrows()):
        product_key = f"{row['Market_Segment']}_{row['Panel_Size']}"
        products.append(product_key)
        
        # Convert to INTEGER weekly demand
        # Round to nearest integer for demand
        weekly_demand = int(round(row['Forecasted_Demand'] / 30 * planning_days))
        
        product_info.append({
            'index': idx,
            'key': product_key,
            'segment': row['Market_Segment'],
            'panel_size': row['Panel_Size'],
            'weekly_demand': weekly_demand,  # INTEGER
            'unit_value': product_values[product_key]  # INTEGER from class method
        })
    
    n_products = len(products)
    
    print(f"\nModel Setup:")
    print(f"  Plants: {n_plants}")
    print(f"  Products: {n_products}")
    print(f"  Planning Days: {planning_days}")
    
    # Calculate total capacity and demand as INTEGERS
    total_weekly_demand = sum(p['weekly_demand'] for p in product_info)
    total_weekly_capacity = sum(
        int(cap['allocated_capacity'] / 30 * planning_days)
        for cap in capacity_allocation.values()
    )
    
    print(f"\nInteger Capacity Check:")
    print(f"  Weekly Demand: {total_weekly_demand:,d} units")
    print(f"  Weekly Capacity: {total_weekly_capacity:,d} units")
    print(f"  Coverage: {(total_weekly_capacity/max(total_weekly_demand,1))*100:.1f}%")
    
    # Scale demand if necessary
    demand_scale = 1.0
    if total_weekly_capacity < total_weekly_demand:
        demand_scale = total_weekly_capacity / total_weekly_demand * 0.95
        print(f"  ⚠️ Scaling demand by {demand_scale:.2%} to ensure feasibility")
        
        # Update product_info with scaled INTEGER demand
        for p in product_info:
            p['weekly_demand'] = max(1, int(p['weekly_demand'] * demand_scale))
    
    # Create CP model
    model = cp_model.CpModel()
    
    # Decision variables - ALL INTEGER
    allocation = {}
    plant_active = {}
    
    print(f"\nCreating integer variables...")
    
    # Create allocation variables
    for p in range(n_products):
        allocation[p] = {}
        panel_size = product_info[p]['panel_size']
        product_demand = product_info[p]['weekly_demand']
        
        for i in range(n_plants):
            plant_name = plants[i]
            plant_row = plant_df[plant_df['Plant_Name'] == plant_name].iloc[0]
            
            # Check capability
            col_name = f"Can_Produce_{panel_size.replace('.', '_').replace('\"', 'in')}"
            can_produce = plant_row[col_name] if col_name in plant_row.index else False
            
            if can_produce:
                # INTEGER capacity
                weekly_capacity = int(capacity_allocation[plant_name]['allocated_capacity'] / 30 * planning_days)
                max_alloc = min(int(weekly_capacity * 0.4), product_demand)
                
                # Only create variable if meaningful production possible
                if max_alloc >= 1:
                    allocation[p][i] = model.NewIntVar(0, max_alloc, f'alloc_{p}_{i}')
                else:
                    allocation[p][i] = model.NewIntVar(0, 0, f'alloc_{p}_{i}_zero')
            else:
                allocation[p][i] = model.NewIntVar(0, 0, f'alloc_{p}_{i}_cannot')
    
    # Plant activation variables
    for i in range(n_plants):
        plant_active[i] = model.NewBoolVar(f'plant_active_{i}')
    
    # Add constraints
    print(f"\nAdding integer constraints...")
    
    # 1. Plant activation constraints
    for i in range(n_plants):
        plant_name = plants[i]
        weekly_capacity = int(capacity_allocation[plant_name]['allocated_capacity'] / 30 * planning_days)
        
        if weekly_capacity < 1:
            model.Add(plant_active[i] == 0)
            continue
        
        total_plant_production = sum(
            allocation[p][i] for p in range(n_products) if i in allocation[p]
        )
        
        # Link activation to production
        model.Add(total_plant_production <= weekly_capacity * plant_active[i])
        
        # Minimum utilization if active
        min_production = max(1, int(weekly_capacity * 0.2))
        model.Add(total_plant_production >= min_production * plant_active[i])
    
    # 2. Demand satisfaction with INTEGER slack
    slack_vars = []
    for p in range(n_products):
        demand = product_info[p]['weekly_demand']  # INTEGER
        
        slack = model.NewIntVar(0, demand, f'slack_{p}')
        slack_vars.append(slack)
        
        total_product_allocation = sum(
            allocation[p][i] for i in range(n_plants) if i in allocation[p]
        )
        model.Add(total_product_allocation + slack == demand)
        
        # Minimum fill rate constraint
        min_fill = int(demand * 0.5)  # At least 50% fill
        model.Add(total_product_allocation >= min_fill)
    
    # 3. Capacity constraints
    for i in range(n_plants):
        plant_name = plants[i]
        weekly_capacity = int(capacity_allocation[plant_name]['allocated_capacity'] / 30 * planning_days)
        
        if weekly_capacity >= 1:
            total_plant_allocation = sum(
                allocation[p][i] for p in range(n_products) if i in allocation[p]
            )
            model.Add(total_plant_allocation <= weekly_capacity)
    
    # Build objective with INTEGER coefficients
    print(f"\nBuilding integer economic objective...")
    
    # Revenue terms
    revenue_terms = []
    for p in range(n_products):
        unit_value = product_info[p]['unit_value']  # INTEGER
        for i in range(n_plants):
            if i in allocation[p]:
                revenue_terms.append(allocation[p][i] * unit_value)
    
    # Variable cost terms
    variable_cost_terms = []
    for p in range(n_products):
        for i in range(n_plants):
            if i in allocation[p]:
                plant_name = plants[i]
                plant_row = plant_df[plant_df['Plant_Name'] == plant_name].iloc[0]
                var_cost = int(plant_row['Total_Cost_Per_Unit'])  # INTEGER
                variable_cost_terms.append(allocation[p][i] * var_cost)
    
    # Fixed cost terms
    fixed_cost_terms = []
    for i in range(n_plants):
        plant_name = plants[i]
        weekly_fixed = fixed_costs[plant_name]['total']  # INTEGER from class method
        fixed_cost_terms.append(plant_active[i] * weekly_fixed)
    
    # Opportunity cost terms
    opportunity_cost_terms = []
    for p in range(n_products):
        lost_value = int(product_info[p]['unit_value'] * 0.5)  # INTEGER
        opportunity_cost_terms.append(slack_vars[p] * lost_value)
    
    # Maximize profit
    model.Maximize(
        sum(revenue_terms) - 
        sum(variable_cost_terms) - 
        sum(fixed_cost_terms) - 
        sum(opportunity_cost_terms)
    )
    
    # Solve
    print(f"\nSolving integer optimization model...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 4
    
    status = solver.Solve(model)
    
    print(f"\nSolver Status: {solver.StatusName(status)}")
    
    # Process results ensuring INTEGER values
    allocations = []
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"✅ Solution found!")
        
        # Calculate economic summary with INTEGER values
        total_revenue = 0
        total_variable_cost = 0
        total_fixed_cost = 0
        total_opportunity_cost = 0
        
        for p in range(n_products):
            for i in range(n_plants):
                if i in allocation[p]:
                    produced = solver.Value(allocation[p][i])
                    if produced > 0:
                        total_revenue += produced * product_info[p]['unit_value']
                        plant_row = plant_df[plant_df['Plant_Name'] == plants[i]].iloc[0]
                        total_variable_cost += produced * int(plant_row['Total_Cost_Per_Unit'])
        
        for i in range(n_plants):
            if solver.Value(plant_active[i]):
                total_fixed_cost += fixed_costs[plants[i]]['total']
        
        for p in range(n_products):
            slack = solver.Value(slack_vars[p])
            total_opportunity_cost += slack * int(product_info[p]['unit_value'] * 0.5)
        
        net_profit = total_revenue - total_variable_cost - total_fixed_cost - total_opportunity_cost
        
        print(f"\n📊 ECONOMIC SUMMARY (All Integer):")
        print(f"  Revenue:           ${total_revenue:,d}")
        print(f"  Variable Costs:   -${total_variable_cost:,d}")
        print(f"  Fixed Costs:      -${total_fixed_cost:,d}")
        print(f"  Opportunity Cost: -${total_opportunity_cost:,d}")
        print(f"  ─────────────────────────────")
        print(f"  NET PROFIT:        ${net_profit:,d}")
        
        # Create allocation records with INTEGER values
        for p in range(n_products):
            info = product_info[p]
            
            # Get INTEGER allocations
            segment_allocations = {}
            total_allocated = 0
            
            for i in range(n_plants):
                if i in allocation[p]:
                    amount = solver.Value(allocation[p][i])
                    
                    # Only include if at least 1 unit
                    if amount >= 1:
                        segment_allocations[plants[i]] = int(amount)
                        total_allocated += int(amount)
            
            # Calculate INTEGER unmet demand
            weekly_demand = info['weekly_demand']  # Already INTEGER
            unmet_demand = int(max(0, weekly_demand - total_allocated))  # Ensure INTEGER
            
            # Fill rate can be float (percentage)
            fill_rate = (total_allocated / weekly_demand * 100) if weekly_demand > 0 else 0.0
            
            allocations.append({
                'Segment': info['segment'],
                'Panel_Size': info['panel_size'],
                'Total_Demand': int(weekly_demand),  # Explicit INTEGER
                'Allocations': segment_allocations,  # Dict of INTEGERS
                'Unmet_Demand': int(unmet_demand),  # Explicit INTEGER
                'Fill_Rate': fill_rate,  # Can be float
                'Value_Created': int(total_allocated * info['unit_value'])  # INTEGER
            })
    
    else:
        print(f"❌ No feasible solution. Using integer heuristic.")
        allocations = integer_heuristic_allocation(
            capacity_allocation, agg_demand, plant_df, fixed_costs, 
            product_values, planning_days
        )
    
    # Validate all values are integers using class method
    validation_passed, issues = RealisticEconomicsModel.validate_integer_allocations(allocations)
    
    if validation_passed:
        print(f"\n✅ VALIDATION PASSED: All quantities are integers")
    else:
        print(f"\n❌ VALIDATION FAILED: Found non-integer values:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
    
    return allocations


def integer_heuristic_allocation(capacity_allocation, agg_demand, plant_df, 
                                fixed_costs, product_values, planning_days):
    """
    Fallback heuristic with guaranteed INTEGER allocations
    """
    print("\n" + "="*80)
    print("INTEGER HEURISTIC ALLOCATION")
    print("="*80)
    
    allocations = []
    
    for _, demand_row in agg_demand.iterrows():
        segment = demand_row['Market_Segment']
        panel_size = demand_row['Panel_Size']
        product_key = f"{segment}_{panel_size}"
        
        # INTEGER demand
        weekly_demand = int(round(demand_row['Forecasted_Demand'] / 30 * planning_days))
        
        # Find capable plants
        capable_plants = []
        for plant_name in capacity_allocation:
            plant_row = plant_df[plant_df['Plant_Name'] == plant_name].iloc[0]
            col_name = f"Can_Produce_{panel_size.replace('.', '_').replace('\"', 'in')}"
            if col_name in plant_row.index and plant_row[col_name]:
                capable_plants.append(plant_name)
        
        segment_allocations = {}
        
        if capable_plants:
            # INTEGER division among plants
            per_plant = weekly_demand // len(capable_plants)
            remainder = weekly_demand % len(capable_plants)
            
            for i, plant_name in enumerate(capable_plants):
                # Distribute remainder to first plants
                allocation = per_plant + (1 if i < remainder else 0)
                
                # Only allocate if at least 1 unit
                if allocation >= 1:
                    segment_allocations[plant_name] = int(allocation)
        
        # Calculate INTEGER totals
        total_allocated = sum(segment_allocations.values())
        unmet_demand = int(max(0, weekly_demand - total_allocated))
        
        allocations.append({
            'Segment': segment,
            'Panel_Size': panel_size,
            'Total_Demand': int(weekly_demand),
            'Allocations': segment_allocations,
            'Unmet_Demand': int(unmet_demand),
            'Fill_Rate': (total_allocated / weekly_demand * 100) if weekly_demand > 0 else 0.0,
            'Value_Created': int(total_allocated * product_values.get(product_key, 500))
        })
    
    return allocations






if __name__ == "__main__":
    print("="*100)
    print("H2OPT WITH DEMAND SMOOTHING DEMONSTRATION")
    print("="*100)
    
    optimizer = H2Opt(
        demand_file='tft_lcd_demand_data.csv',
        plant_file='tft_lcd_plant_data.csv'
    )
    

    print("="*100)
    print("CONSISTENT INTEGER OPTIMIZATION WITH CLASS STRUCTURE")
    print("="*100)
    print("\n✅ Using RealisticEconomicsModel class methods:")
    print("   - calculate_fixed_costs()")
    print("   - calculate_production_value()")
    print("   - validate_integer_allocations()")
    print("\nThis maintains consistency with h2_opt.py implementation")
    # results = optimizer.run_optimization()
    
    print("\n" + "="*100)
    print("KEY INNOVATIONS")
    print("="*100)
    
    print("\n📊 DEMAND SMOOTHING BENEFITS:")
    print("1. ✓ Shifts low-priority orders to balance capacity")
    print("2. ✓ Respects lead time constraints")
    print("3. ✓ Maintains high-priority order dates")
    print("4. ✓ Reduces demand peaks that exceed capacity")
    print("5. ✓ Improves overall system feasibility")
    
    print("\n🎯 RESULT:")
    print(f"   Fill Rate Achieved: {results['metrics']['fill_rate']:.1f}%")
    print(f"   Without overwhelming any single period!")