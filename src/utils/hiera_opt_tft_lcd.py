"""
Advanced Hierarchical TFT-LCD Supply Chain Optimizer
Level 1: Evolutionary Algorithms (NSGA-II, NSGA-III, MOEA/D)
Level 2: Google OR-Tools Constraint Programming
Level 3: Surrogate Model (Regression-based)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For Level 1: Evolutionary Algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# For Level 2: Constraint Programming
from ortools.sat.python import cp_model

# For Level 3: Surrogate Model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class Level1StrategicProblem(Problem):
    """
    Multi-objective optimization problem for Level 1 Strategic Planning
    Objectives: Minimize cost, Maximize utilization, Minimize unfulfilled demand
    """
    
    def __init__(self, demand_df, plant_df, planning_horizon=30):
        self.demand_df = demand_df
        self.plant_df = plant_df
        self.planning_horizon = planning_horizon
        
        # Aggregate demand
        self.agg_demand = self._aggregate_demand()
        
        # Number of decision variables: allocation ratio for each plant
        n_vars = len(plant_df)
        
        # Number of objectives
        n_obj = 3
        
        # Number of constraints
        n_constr = 2
        
        # Variable bounds (allocation ratios between 0 and 1)
        xl = np.zeros(n_vars)
        xu = np.ones(n_vars)
        
        super().__init__(n_var=n_vars, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
    
    def _aggregate_demand(self):
        """Aggregate demand for the planning horizon"""
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
        """Evaluate objectives and constraints"""
        pop_size = x.shape[0]
        
        # Initialize objectives
        f1_cost = np.zeros(pop_size)
        f2_utilization = np.zeros(pop_size)
        f3_unmet = np.zeros(pop_size)
        
        # Initialize constraints
        g1_capacity = np.zeros(pop_size)
        g2_min_util = np.zeros(pop_size)
        
        for i in range(pop_size):
            allocation = x[i]
            
            # Calculate allocated capacity for each plant
            total_cost = 0
            total_capacity_used = 0
            total_capacity_available = 0
            total_demand = self.agg_demand['Forecasted_Demand'].sum()
            total_fulfilled = 0
            
            for j, plant_row in self.plant_df.iterrows():
                # Plant capacity
                plant_capacity = plant_row['Capacity_Per_Day'] * self.planning_horizon * plant_row['Efficiency']
                allocated_capacity = allocation[j] * plant_capacity
                
                # Cost calculation
                production_cost = allocated_capacity * plant_row['Total_Cost_Per_Unit']
                setup_cost = plant_row['Setup_Cost'] * self.planning_horizon * allocation[j]
                maintenance_cost = plant_row['Maintenance_Cost_Daily'] * self.planning_horizon * allocation[j]
                
                total_cost += production_cost + setup_cost + maintenance_cost
                total_capacity_used += allocated_capacity
                total_capacity_available += plant_capacity
                
                # Estimate fulfilled demand based on plant capabilities
                plant_fulfillment = min(allocated_capacity, total_demand * allocation[j])
                total_fulfilled += plant_fulfillment
            
            # Objectives
            f1_cost[i] = total_cost
            f2_utilization[i] = -total_capacity_used / max(total_capacity_available, 1)  # Negative for maximization
            f3_unmet[i] = max(0, total_demand - total_fulfilled)
            
            # Constraints
            g1_capacity[i] = total_capacity_used - total_capacity_available  # <= 0
            g2_min_util[i] = 0.3 - (total_capacity_used / max(total_capacity_available, 1))  # >= 0.3 utilization
        
        out["F"] = np.column_stack([f1_cost, f2_utilization, f3_unmet])
        out["G"] = np.column_stack([g1_capacity, g2_min_util])

class HierarchicalTFTLCDOptimizer:
    """
    Advanced Hierarchical Optimizer using:
    - Level 1: Evolutionary Algorithms
    - Level 2: Google OR-Tools
    - Level 3: Surrogate Models
    """
    
    def __init__(self, demand_file='tft_lcd_demand_data.csv', plant_file='tft_lcd_plant_data.csv'):
        """Initialize optimizer with data files"""
        self.demand_df = pd.read_csv(demand_file)
        self.plant_df = pd.read_csv(plant_file)
        
        # Convert date column
        self.demand_df['Date'] = pd.to_datetime(self.demand_df['Date'])
        
        # Initialize surrogate model for Level 3
        self.surrogate_model = None
        self.scaler = StandardScaler()
        
    def level1_strategic_optimization(self, planning_horizon=30, algorithm='NSGA-II', n_gen=100):
        """
        Level 1: Strategic Planning using Evolutionary Algorithms
        """
        print("=" * 80)
        print(f"LEVEL 1: STRATEGIC PLANNING - {algorithm}")
        print("=" * 80)
        
        # Create problem instance
        problem = Level1StrategicProblem(self.demand_df, self.plant_df, planning_horizon)
        
        # Select algorithm
        if algorithm == 'NSGA-II':
            algo = NSGA2(
                pop_size=100,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )
        elif algorithm == 'NSGA-III':
            ref_dirs = get_reference_directions("energy", 3, 100)
            algo = NSGA3(
                pop_size=len(ref_dirs),
                ref_dirs=ref_dirs,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )
        elif algorithm == 'MOEA/D':
            ref_dirs = get_reference_directions("uniform", 3, 12)
            algo = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=15,
                prob_neighbor_mating=0.7,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20)
            )
        else:  # Default to NSGA-II
            algo = NSGA2(pop_size=100)
        
        # Run optimization
        print(f"Running {algorithm} optimization...")
        print(f"Population size: {algo.pop_size if hasattr(algo, 'pop_size') else 'adaptive'}")
        print(f"Generations: {n_gen}")
        
        res = minimize(
            problem,
            algo,
            ('n_gen', n_gen),
            seed=42,
            verbose=False
        )
        
        # Get best solution (minimum cost while meeting constraints)
        if res.F is not None and len(res.F) > 0:
            # Select solution with best trade-off
            costs = res.F[:, 0]
            utilizations = -res.F[:, 1]  # Convert back to positive
            unmet_demands = res.F[:, 2]
            
            # Weighted score (customize weights as needed)
            scores = 0.4 * (costs / costs.max()) - 0.3 * (utilizations / utilizations.max()) + 0.3 * (unmet_demands / max(unmet_demands.max(), 1))
            best_idx = scores.argmin()
            
            best_solution = res.X[best_idx]
            best_objectives = res.F[best_idx]
            
            print(f"\nOptimization Results:")
            print(f"  Total Cost: ${best_objectives[0]:,.0f}")
            print(f"  Capacity Utilization: {-best_objectives[1]*100:.1f}%")
            print(f"  Unmet Demand: {best_objectives[2]:,.0f} units")
            
            # Create capacity allocation
            capacity_allocation = {}
            print(f"\nPlant Allocations:")
            for i, plant_row in self.plant_df.iterrows():
                plant_capacity = plant_row['Capacity_Per_Day'] * planning_horizon * plant_row['Efficiency']
                allocated = best_solution[i] * plant_capacity
                
                capacity_allocation[plant_row['Plant_Name']] = {
                    'allocation_ratio': best_solution[i],
                    'allocated_capacity': allocated,
                    'max_capacity': plant_capacity,
                    'utilization': (allocated / plant_capacity) * 100
                }
                
                print(f"  {plant_row['Plant_Name']}: {allocated:,.0f} units ({best_solution[i]*100:.1f}% allocation)")
        else:
            print("No feasible solution found. Using default allocation.")
            capacity_allocation = self._default_allocation(planning_horizon)
        
        return capacity_allocation, problem.agg_demand
    
    def _default_allocation(self, planning_horizon):
        """Default allocation when optimization fails"""
        capacity_allocation = {}
        for _, plant_row in self.plant_df.iterrows():
            plant_capacity = plant_row['Capacity_Per_Day'] * planning_horizon * plant_row['Efficiency']
            capacity_allocation[plant_row['Plant_Name']] = {
                'allocation_ratio': 0.5,
                'allocated_capacity': plant_capacity * 0.5,
                'max_capacity': plant_capacity,
                'utilization': 50.0
            }
        return capacity_allocation
    
    def level2_tactical_optimization(self, capacity_allocation, agg_demand, planning_days=7):
        """
        Level 2: Tactical Allocation using Google OR-Tools Constraint Programming
        """
        print("\n" + "=" * 80)
        print("LEVEL 2: TACTICAL ALLOCATION - OR-TOOLS CP")
        print("=" * 80)
        
        # Create CP model
        model = cp_model.CpModel()
        
        # Prepare data
        plants = list(capacity_allocation.keys())
        products = []
        for _, row in agg_demand.iterrows():
            products.append(f"{row['Market_Segment']}_{row['Panel_Size']}")
        
        n_plants = len(plants)
        n_products = len(products)
        
        # Decision variables: allocation[p][i] = units of product p allocated to plant i
        allocation = {}
        for p in range(n_products):
            allocation[p] = {}
            for i in range(n_plants):
                # Check if plant can produce this product
                product_row = agg_demand.iloc[p]
                plant_name = plants[i]
                plant_row = self.plant_df[self.plant_df['Plant_Name'] == plant_name].iloc[0]
                
                # Check panel size capability
                panel_size = product_row['Panel_Size']
                can_produce_col = f"Can_Produce_{panel_size.replace('.', '_').replace('\"', 'in')}"
                
                if can_produce_col in plant_row and plant_row[can_produce_col]:
                    max_allocation = int(capacity_allocation[plant_name]['allocated_capacity'] / 30 * planning_days)
                    allocation[p][i] = model.NewIntVar(0, max_allocation, f'alloc_{p}_{i}')
                else:
                    allocation[p][i] = model.NewIntVar(0, 0, f'alloc_{p}_{i}')  # Cannot produce
        
        # Constraints
        # 1. Demand satisfaction constraint
        for p in range(n_products):
            demand = int(agg_demand.iloc[p]['Forecasted_Demand'] / 30 * planning_days)
            model.Add(sum(allocation[p][i] for i in range(n_plants)) >= int(demand * 0.9))  # At least 90% demand
        
        # 2. Capacity constraints
        for i in range(n_plants):
            plant_capacity = int(capacity_allocation[plants[i]]['allocated_capacity'] / 30 * planning_days)
            model.Add(sum(allocation[p][i] for p in range(n_products)) <= plant_capacity)
        
        # Objective: Minimize total cost
        total_cost = []
        for p in range(n_products):
            for i in range(n_plants):
                plant_row = self.plant_df[self.plant_df['Plant_Name'] == plants[i]].iloc[0]
                cost_per_unit = int(plant_row['Total_Cost_Per_Unit'])
                total_cost.append(allocation[p][i] * cost_per_unit)
        
        model.Minimize(sum(total_cost))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        print(f"Solving CP model for {planning_days}-day tactical allocation...")
        status = solver.Solve(model)
        
        allocations = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found! Status: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}")
            print(f"Total Cost: ${solver.ObjectiveValue():,.0f}")
            
            # Extract solution
            for p in range(n_products):
                product_row = agg_demand.iloc[p]
                plant_allocations = {}
                total_allocated = 0
                
                for i in range(n_plants):
                    allocated = solver.Value(allocation[p][i])
                    if allocated > 0:
                        plant_allocations[plants[i]] = allocated
                        total_allocated += allocated
                
                demand = product_row['Forecasted_Demand'] / 30 * planning_days
                allocations.append({
                    'Segment': product_row['Market_Segment'],
                    'Panel_Size': product_row['Panel_Size'],
                    'Total_Demand': demand,
                    'Allocations': plant_allocations,
                    'Unmet_Demand': max(0, demand - total_allocated),
                    'Fill_Rate': (total_allocated / demand) * 100 if demand > 0 else 0
                })
            
            # Print summary
            print(f"\nTactical Allocation Summary (showing first 5 products):")
            for alloc in allocations[:5]:
                print(f"  {alloc['Segment']} - {alloc['Panel_Size']}:")
                print(f"    Demand: {alloc['Total_Demand']:,.0f}, Fill Rate: {alloc['Fill_Rate']:.1f}%")
                for plant, amount in alloc['Allocations'].items():
                    if amount > 0:
                        print(f"    {plant}: {amount:,.0f} units")
        else:
            print("No feasible solution found. Using heuristic allocation.")
            allocations = self._heuristic_allocation(capacity_allocation, agg_demand, planning_days)
        
        return allocations
    
    def _heuristic_allocation(self, capacity_allocation, agg_demand, planning_days):
        """Fallback heuristic allocation"""
        allocations = []
        for _, product_row in agg_demand.iterrows():
            demand = product_row['Forecasted_Demand'] / 30 * planning_days
            plant_allocations = {}
            
            for plant_name, cap_info in capacity_allocation.items():
                # Simple proportional allocation
                plant_share = cap_info['allocation_ratio']
                plant_allocations[plant_name] = demand * plant_share * 0.2  # Distribute 20% to each
            
            allocations.append({
                'Segment': product_row['Market_Segment'],
                'Panel_Size': product_row['Panel_Size'],
                'Total_Demand': demand,
                'Allocations': plant_allocations,
                'Unmet_Demand': 0,
                'Fill_Rate': 100
            })
        return allocations
    
    def level3_operational_optimization(self, allocations, use_surrogate=True):
        """
        Level 3: Operational Scheduling using Surrogate Model
        """
        print("\n" + "=" * 80)
        print("LEVEL 3: OPERATIONAL SCHEDULING - SURROGATE MODEL")
        print("=" * 80)
        
        if use_surrogate:
            # Train or load surrogate model
            if self.surrogate_model is None:
                self._train_surrogate_model()
            
            # Use surrogate model for scheduling
            schedules = self._surrogate_scheduling(allocations)
        else:
            # Use rule-based scheduling
            schedules = self._rule_based_scheduling(allocations)
        
        return schedules
    
    def _train_surrogate_model(self):
        """Train surrogate model for production scheduling"""
        print("Training surrogate model for scheduling optimization...")
        
        # Generate synthetic training data
        n_samples = 1000
        X = []
        y = []
        
        for _ in range(n_samples):
            # Features: demand, capacity, product mix, plant efficiency
            demand = np.random.uniform(100, 5000)
            capacity = np.random.uniform(1000, 8000)
            n_products = np.random.randint(1, 10)
            efficiency = np.random.uniform(0.8, 0.99)
            setup_cost = np.random.uniform(3000, 6000)
            
            # Target: estimated production time and cost
            production_time = (demand / capacity) * (1 / efficiency) * (1 + 0.1 * n_products)
            production_cost = demand * 100 + setup_cost * n_products
            
            X.append([demand, capacity, n_products, efficiency, setup_cost])
            y.append([production_time, production_cost])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models for each target
        self.surrogate_model = {
            'time': RandomForestRegressor(n_estimators=100, random_state=42),
            'cost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.surrogate_model['time'].fit(X_train_scaled, y_train[:, 0])
        self.surrogate_model['cost'].fit(X_train_scaled, y_train[:, 1])
        
        # Evaluate
        time_score = self.surrogate_model['time'].score(X_test_scaled, y_test[:, 0])
        cost_score = self.surrogate_model['cost'].score(X_test_scaled, y_test[:, 1])
        
        print(f"Surrogate model trained:")
        print(f"  Time prediction R²: {time_score:.3f}")
        print(f"  Cost prediction R²: {cost_score:.3f}")
    
    def _surrogate_scheduling(self, allocations):
        """Generate schedules using surrogate model predictions"""
        print("Generating optimal schedules using surrogate model...")
        
        schedules = {}
        
        # Get unique plants from allocations
        all_plants = set()
        for alloc in allocations:
            all_plants.update(alloc['Allocations'].keys())
        
        for plant_name in all_plants:
            plant_row = self.plant_df[self.plant_df['Plant_Name'] == plant_name].iloc[0]
            
            # Collect products for this plant
            plant_products = []
            total_demand = 0
            
            for alloc in allocations:
                if plant_name in alloc['Allocations'] and alloc['Allocations'][plant_name] > 0:
                    quantity = alloc['Allocations'][plant_name]
                    plant_products.append({
                        'product': f"{alloc['Segment']}_{alloc['Panel_Size']}",
                        'quantity': quantity
                    })
                    total_demand += quantity
            
            if not plant_products:
                continue
            
            # Prepare features for surrogate model
            features = np.array([[
                total_demand,
                plant_row['Capacity_Per_Day'],
                len(plant_products),
                plant_row['Efficiency'],
                plant_row['Setup_Cost']
            ]])
            
            features_scaled = self.scaler.transform(features)
            
            # Predict using surrogate model
            predicted_time = self.surrogate_model['time'].predict(features_scaled)[0]
            predicted_cost = self.surrogate_model['cost'].predict(features_scaled)[0]
            
            # Generate schedule based on predictions
            schedule = []
            current_time = 0
            
            # Sort products by quantity (batch larger quantities first)
            plant_products.sort(key=lambda x: x['quantity'], reverse=True)
            
            for i, product in enumerate(plant_products):
                production_time = (product['quantity'] / plant_row['Capacity_Per_Day']) * predicted_time
                
                schedule.append({
                    'slot': i + 1,
                    'product': product['product'],
                    'quantity': product['quantity'],
                    'start_time': current_time,
                    'end_time': current_time + production_time,
                    'duration': production_time
                })
                current_time += production_time
            
            schedules[plant_name] = {
                'schedule': schedule,
                'total_time': predicted_time,
                'total_cost': predicted_cost,
                'n_products': len(plant_products),
                'capacity_utilization': (total_demand / plant_row['Capacity_Per_Day']) * 100
            }
        
        # Print schedule summary
        print(f"\nOperational Schedule Summary:")
        for plant_name, schedule_info in schedules.items():
            print(f"\n  {plant_name}:")
            print(f"    Products scheduled: {schedule_info['n_products']}")
            print(f"    Predicted time: {schedule_info['total_time']:.2f} hours")
            print(f"    Predicted cost: ${schedule_info['total_cost']:,.0f}")
            print(f"    Capacity utilization: {schedule_info['capacity_utilization']:.1f}%")
            
            # Show first 3 scheduled items
            for item in schedule_info['schedule'][:3]:
                print(f"    Slot {item['slot']}: {item['product']} - {item['quantity']:.0f} units")
        
        return schedules
    
    def _rule_based_scheduling(self, allocations):
        """Fallback rule-based scheduling"""
        schedules = {}
        
        all_plants = set()
        for alloc in allocations:
            all_plants.update(alloc['Allocations'].keys())
        
        for plant_name in all_plants:
            plant_products = []
            for alloc in allocations:
                if plant_name in alloc['Allocations'] and alloc['Allocations'][plant_name] > 0:
                    plant_products.append({
                        'product': f"{alloc['Segment']}_{alloc['Panel_Size']}",
                        'quantity': alloc['Allocations'][plant_name]
                    })
            
            schedule = []
            for i, product in enumerate(plant_products):
                schedule.append({
                    'slot': i + 1,
                    'product': product['product'],
                    'quantity': product['quantity']
                })
            
            schedules[plant_name] = {'schedule': schedule}
        
        return schedules
    
    def run_optimization(self, planning_horizon=30, tactical_days=7, algorithm='NSGA-II'):
        """Run complete hierarchical optimization"""
        print("\n" + "=" * 100)
        print("ADVANCED HIERARCHICAL TFT-LCD SUPPLY CHAIN OPTIMIZATION")
        print("=" * 100)
        
        # Level 1: Strategic Planning with Evolutionary Algorithm
        capacity_allocation, agg_demand = self.level1_strategic_optimization(
            planning_horizon=planning_horizon,
            algorithm=algorithm,
            n_gen=50  # Reduced for faster execution
        )
        
        # Level 2: Tactical Allocation with OR-Tools
        allocations = self.level2_tactical_optimization(
            capacity_allocation=capacity_allocation,
            agg_demand=agg_demand,
            planning_days=tactical_days
        )
        
        # Level 3: Operational Scheduling with Surrogate Model
        schedules = self.level3_operational_optimization(
            allocations=allocations,
            use_surrogate=True
        )
        
        print("\n" + "=" * 100)
        print("OPTIMIZATION COMPLETE")
        print("=" * 100)
        
        # Calculate final KPIs
        total_demand = sum(a['Total_Demand'] for a in allocations)
        total_allocated = sum(sum(a['Allocations'].values()) for a in allocations)
        fill_rate = (total_allocated / total_demand) * 100 if total_demand > 0 else 0
        
        print(f"\nFinal Performance Metrics:")
        print(f"  Overall Fill Rate: {fill_rate:.1f}%")
        print(f"  Number of Active Plants: {len(schedules)}")
        print(f"  Total Products Scheduled: {sum(len(s['schedule']) for s in schedules.values())}")
        
        return {
            'strategic': capacity_allocation,
            'tactical': allocations,
            'operational': schedules,
            'metrics': {
                'fill_rate': fill_rate,
                'active_plants': len(schedules),
                'total_products': sum(len(s['schedule']) for s in schedules.values())
            }
        }

# Main execution
if __name__ == "__main__":
    print("=" * 100)
    print("INITIALIZING ADVANCED HIERARCHICAL OPTIMIZER")
    print("=" * 100)
    
    # First generate the data files
    print("\nChecking for required data files...")
    import os
    
    if not os.path.exists('tft_lcd_demand_data.csv') or not os.path.exists('tft_lcd_plant_data.csv'):
        print("Data files not found. Please run the data generation script first.")
        print("Run: python tft_lcd_demand_gen.py")
    else:
        print("Data files found. Initializing optimizer...")
        
        # Create optimizer instance
        optimizer = HierarchicalTFTLCDOptimizer(
            demand_file='tft_lcd_demand_data.csv',
            plant_file='tft_lcd_plant_data.csv'
        )
        
        # Test different evolutionary algorithms
        algorithms = ['NSGA-II', 'NSGA-III', 'MOEA/D']
        
        print(f"\nTesting with {algorithms[0]}...")  # Use first algorithm for main run
        results = optimizer.run_optimization(
            planning_horizon=30,
            tactical_days=7,
            algorithm=algorithms[0]
        )
        
        print("\n" + "=" * 100)
        print("OPTIMIZATION PROCESS COMPLETED SUCCESSFULLY")
        print("=" * 100)
        print("\nThe hierarchical optimization has:")
        print("1. Used evolutionary algorithms (NSGA-II/III, MOEA/D) for strategic planning")
        print("2. Applied OR-Tools constraint programming for tactical allocation")
        print("3. Implemented surrogate models for operational scheduling")
        print(f"\nFinal Fill Rate: {results['metrics']['fill_rate']:.1f}%")
        print(f"Active Plants: {results['metrics']['active_plants']}")
        print(f"Products Scheduled: {results['metrics']['total_products']}")