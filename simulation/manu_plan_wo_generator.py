"""
Manufacturing Work Order Generator
Integrates with existing SQLAlchemy models and simulation system
"""

from datetime import datetime, timedelta, date
from decimal import Decimal
import random
import string
from typing import List, Optional, Dict, Any
from enum import Enum

# Import from your existing datamodel
from manufacturing.datamodel import (
    get_session, create_engine,
    WorkOrder, WorkOrderStatus, 
    Product, Route, Sequence, Operation,
    Equipment, EquipmentStatus,
    WIPRecord, WIPStatus,
    Material, MaterialType,
    BOMItem, UnitOfMeasure
)

class PriorityLevel(Enum):
    """Work order priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ROUTINE = 5

class WorkOrderGenerator:
    """Generate work orders for manufacturing simulation"""
    
    def __init__(self, session):
        """
        Initialize generator with database session
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        self.order_counter = self._get_next_order_number()
        
        # Load existing entities from database
        self.products = []
        self.routes = []
        self.equipment = []
        self._load_entities()
        
        # Customer pool for realistic data
        self.customers = [
            "Automotive Corp", "TechPro Industries", "Global Manufacturing",
            "Precision Parts Ltd", "Advanced Systems Inc", "Quality Components",
            "Industrial Solutions", "Production Partners", "Supply Chain Co",
            "Manufacturing Excellence"
        ]
        
        # Sales order prefixes
        self.sales_prefixes = ["SO", "PO", "ORD", "REQ"]
        
    def _load_entities(self):
        """Load existing entities from database"""
        # Load active products with routes
        self.products = self.session.query(Product).filter(
            Product.is_active == True
        ).all()
        
        # Load active routes
        self.routes = self.session.query(Route).filter(
            Route.is_active == True
        ).all()
        
        # Load available equipment
        self.equipment = self.session.query(Equipment).filter(
            Equipment.status.in_([EquipmentStatus.AVAILABLE, EquipmentStatus.IN_USE])
        ).all()
        
        print(f"Loaded {len(self.products)} products, {len(self.routes)} routes, {len(self.equipment)} equipment")
        
    def _get_next_order_number(self) -> int:
        """Get next available work order number"""
        last_order = self.session.query(WorkOrder).order_by(
            WorkOrder.id.desc()
        ).first()
        
        if last_order:
            # Extract number from last order
            try:
                parts = last_order.work_order_number.split('-')
                if len(parts) >= 3:
                    return int(parts[-1]) + 1
            except:
                pass
        return 1000
    
    def generate_work_order(self, 
                          product: Optional[Product] = None,
                          route: Optional[Route] = None,
                          quantity: Optional[float] = None,
                          due_date: Optional[datetime] = None,
                          priority: Optional[int] = None,
                          status: Optional[WorkOrderStatus] = None,
                          **kwargs) -> WorkOrder:
        """
        Generate a single work order
        
        Args:
            product: Product to manufacture (random if None)
            route: Route to use (primary route if None)
            quantity: Order quantity (random if None)
            due_date: Due date (random future date if None)
            priority: Priority level (random if None)
            status: Initial status (PLANNED if None)
            **kwargs: Additional WorkOrder fields
            
        Returns:
            WorkOrder instance
        """
        
        # Select product
        if product is None:
            if not self.products:
                raise ValueError("No products found in database. Please create products first.")
            product = random.choice(self.products)
        
        # Select route
        if route is None:
            # Get primary route for product
            product_routes = [r for r in self.routes if r.product_id == product.id]
            if product_routes:
                # Prefer primary route
                primary_routes = [r for r in product_routes if r.is_primary]
                route = primary_routes[0] if primary_routes else product_routes[0]
            else:
                # Use any compatible route or None
                route = None
        
        # Generate work order number
        year = datetime.now().year
        wo_number = f"WO-{year}-{self.order_counter:04d}"
        self.order_counter += 1
        
        # Set quantity
        if quantity is None:
            # Random quantity based on typical lot sizes
            quantity = Decimal(random.choice([10, 25, 50, 100, 200, 500, 1000]))
        
        # Set due date
        if due_date is None:
            # Random date between 3 and 30 days from now
            days_ahead = random.randint(3, 30)
            due_date = datetime.now() + timedelta(days=days_ahead)
        
        # Set priority
        if priority is None:
            # Weighted priority selection (more medium priority)
            priority = random.choices(
                [1, 2, 3, 4, 5],
                weights=[5, 15, 50, 20, 10]
            )[0]
        
        # Set status
        if status is None:
            status = WorkOrderStatus.PLANNED
        
        # Set planned dates based on route cycle time
        if route and route.total_cycle_time:
            # Calculate backward from due date
            production_time_hours = float(route.total_cycle_time) * float(quantity) / 60
            planned_end_date = due_date - timedelta(hours=1)  # 1 hour buffer
            planned_start_date = planned_end_date - timedelta(hours=production_time_hours)
        else:
            # Default scheduling
            planned_end_date = due_date - timedelta(hours=1)
            planned_start_date = planned_end_date - timedelta(days=2)
        
        # Generate sales order reference
        sales_order = f"{random.choice(self.sales_prefixes)}-{year}-{random.randint(1000, 9999)}"
        
        # Select customer
        customer = random.choice(self.customers)
        
        # Calculate estimated cost
        estimated_cost = self._calculate_estimated_cost(product, quantity, route)
        
        # Create work order
        work_order = WorkOrder(
            work_order_number=wo_number,
            product=product,
            route=route,
            order_quantity=quantity,
            completed_quantity=Decimal(0),
            scrapped_quantity=Decimal(0),
            priority=priority,
            planned_start_date=planned_start_date,
            planned_end_date=planned_end_date,
            due_date=due_date,
            status=status,
            sales_order_number=sales_order,
            customer_name=customer,
            estimated_cost=estimated_cost,
            notes=kwargs.get('notes', f"Auto-generated order for {customer}"),
            created_by=kwargs.get('created_by', 'WorkOrderGenerator'),
            **{k: v for k, v in kwargs.items() if k not in ['notes', 'created_by']}
        )
        
        return work_order
    
    def _calculate_estimated_cost(self, product: Product, quantity: Decimal, 
                                 route: Optional[Route]) -> Decimal:
        """Calculate estimated cost for work order"""
        base_cost = Decimal(0)
        
        # Material cost from product standard cost
        if product.standard_cost:
            base_cost = product.standard_cost * quantity
        else:
            # Estimate based on product type
            base_cost = Decimal(random.uniform(10, 100)) * quantity
        
        # Add processing cost based on route
        if route and route.total_labor_time:
            labor_rate = Decimal(50)  # $/hour
            labor_cost = (route.total_labor_time / 60) * labor_rate * quantity
            base_cost += labor_cost
        
        # Add overhead (20%)
        base_cost *= Decimal(1.2)
        
        return base_cost.quantize(Decimal('0.01'))
    
    def generate_batch(self, count: int = 10, **kwargs) -> List[WorkOrder]:
        """
        Generate multiple work orders
        
        Args:
            count: Number of work orders to generate
            **kwargs: Default parameters for all orders
            
        Returns:
            List of WorkOrder instances
        """
        work_orders = []
        
        for _ in range(count):
            wo = self.generate_work_order(**kwargs)
            work_orders.append(wo)
            
        return work_orders
    
    def generate_scenario(self, scenario_type: str = 'normal', 
                         duration_days: int = 7) -> Dict[str, Any]:
        """
        Generate work orders for specific scenarios
        
        Args:
            scenario_type: Type of scenario to generate
            duration_days: Duration of scenario in days
            
        Returns:
            Dictionary with scenario details and work orders
        """
        
        scenarios = {
            'normal': self._scenario_normal,
            'high_volume': self._scenario_high_volume,
            'rush_orders': self._scenario_rush_orders,
            'mixed_priority': self._scenario_mixed_priority,
            'capacity_test': self._scenario_capacity_test,
            'bottleneck_test': self._scenario_bottleneck
        }
        
        if scenario_type not in scenarios:
            scenario_type = 'normal'
        
        return scenarios[scenario_type](duration_days)
    
    def _scenario_normal(self, duration_days: int) -> Dict[str, Any]:
        """Normal production scenario"""
        work_orders = []
        start_date = datetime.now()
        
        # Generate orders spread across duration
        orders_per_day = 3
        total_orders = duration_days * orders_per_day
        
        for day in range(duration_days):
            for _ in range(orders_per_day):
                due_date = start_date + timedelta(days=day + random.randint(3, 10))
                wo = self.generate_work_order(
                    due_date=due_date,
                    priority=random.choice([2, 3, 3, 3, 4]),  # Mostly medium
                    status=WorkOrderStatus.PLANNED
                )
                work_orders.append(wo)
        
        return {
            'name': 'Normal Production',
            'description': f'Steady production over {duration_days} days',
            'work_orders': work_orders,
            'metrics': {
                'total_orders': len(work_orders),
                'orders_per_day': orders_per_day,
                'avg_priority': sum(wo.priority for wo in work_orders) / len(work_orders)
            }
        }
    
    def _scenario_high_volume(self, duration_days: int) -> Dict[str, Any]:
        """High volume production scenario"""
        work_orders = []
        
        # Generate many small orders
        orders_count = duration_days * 10
        
        for i in range(orders_count):
            due_date = datetime.now() + timedelta(days=random.randint(2, duration_days + 5))
            wo = self.generate_work_order(
                quantity=Decimal(random.randint(5, 50)),  # Small quantities
                due_date=due_date,
                priority=3,  # All medium priority
                status=WorkOrderStatus.PLANNED
            )
            work_orders.append(wo)
        
        return {
            'name': 'High Volume Production',
            'description': f'Many small orders over {duration_days} days',
            'work_orders': work_orders,
            'metrics': {
                'total_orders': len(work_orders),
                'avg_quantity': sum(float(wo.order_quantity) for wo in work_orders) / len(work_orders),
                'total_units': sum(float(wo.order_quantity) for wo in work_orders)
            }
        }
    
    def _scenario_rush_orders(self, duration_days: int) -> Dict[str, Any]:
        """Rush orders scenario with tight deadlines"""
        work_orders = []
        
        # Mix of normal and rush orders
        for i in range(duration_days * 4):
            is_rush = random.random() < 0.3  # 30% rush orders
            
            if is_rush:
                due_date = datetime.now() + timedelta(days=random.randint(1, 2))
                priority = 1  # Critical
                quantity = Decimal(random.randint(10, 100))
            else:
                due_date = datetime.now() + timedelta(days=random.randint(3, 10))
                priority = random.choice([2, 3, 4])
                quantity = Decimal(random.randint(50, 500))
            
            wo = self.generate_work_order(
                quantity=quantity,
                due_date=due_date,
                priority=priority,
                status=WorkOrderStatus.PLANNED
            )
            work_orders.append(wo)
        
        return {
            'name': 'Rush Orders Mix',
            'description': 'Mix of rush and normal orders',
            'work_orders': work_orders,
            'metrics': {
                'total_orders': len(work_orders),
                'rush_orders': sum(1 for wo in work_orders if wo.priority == 1),
                'avg_lead_time': sum((wo.due_date - datetime.now()).days for wo in work_orders) / len(work_orders)
            }
        }
    
    def _scenario_mixed_priority(self, duration_days: int) -> Dict[str, Any]:
        """Mixed priority scenario for testing scheduling"""
        work_orders = []
        
        priority_distribution = {
            1: 5,   # 5% critical
            2: 15,  # 15% high
            3: 50,  # 50% medium
            4: 20,  # 20% low
            5: 10   # 10% routine
        }
        
        for i in range(duration_days * 5):
            # Select priority based on distribution
            priority = random.choices(
                list(priority_distribution.keys()),
                weights=list(priority_distribution.values())
            )[0]
            
            # Adjust due date based on priority
            if priority == 1:
                due_days = random.randint(1, 3)
            elif priority == 2:
                due_days = random.randint(2, 5)
            else:
                due_days = random.randint(5, 15)
            
            due_date = datetime.now() + timedelta(days=due_days)
            
            wo = self.generate_work_order(
                due_date=due_date,
                priority=priority,
                status=WorkOrderStatus.PLANNED
            )
            work_orders.append(wo)
        
        return {
            'name': 'Mixed Priority',
            'description': 'Realistic priority distribution',
            'work_orders': work_orders,
            'metrics': {
                'total_orders': len(work_orders),
                'priority_distribution': {p: sum(1 for wo in work_orders if wo.priority == p) 
                                        for p in range(1, 6)}
            }
        }
    
    def _scenario_capacity_test(self, duration_days: int) -> Dict[str, Any]:
        """Capacity stress test scenario"""
        work_orders = []
        
        # Calculate theoretical capacity
        total_equipment_hours = len(self.equipment) * duration_days * 8  # 8 hours per day
        
        # Generate orders to utilize 120% of capacity (overload)
        target_hours = total_equipment_hours * 1.2
        accumulated_hours = 0
        
        while accumulated_hours < target_hours:
            wo = self.generate_work_order(
                quantity=Decimal(random.randint(50, 200)),
                due_date=datetime.now() + timedelta(days=random.randint(1, duration_days)),
                priority=random.choice([2, 3]),
                status=WorkOrderStatus.PLANNED
            )
            
            # Estimate processing time
            if wo.route and wo.route.total_cycle_time:
                accumulated_hours += float(wo.route.total_cycle_time * wo.order_quantity / 60)
            else:
                accumulated_hours += float(wo.order_quantity) * 0.1  # Default estimate
            
            work_orders.append(wo)
        
        return {
            'name': 'Capacity Stress Test',
            'description': f'120% capacity utilization over {duration_days} days',
            'work_orders': work_orders,
            'metrics': {
                'total_orders': len(work_orders),
                'theoretical_capacity': total_equipment_hours,
                'planned_hours': accumulated_hours,
                'utilization': (accumulated_hours / total_equipment_hours * 100) if total_equipment_hours > 0 else 0
            }
        }
    
    def _scenario_bottleneck(self, duration_days: int) -> Dict[str, Any]:
        """Scenario to test bottleneck handling"""
        work_orders = []
        
        # Find products that use similar routes (potential bottleneck)
        if self.products:
            # Group products by similar routes
            common_products = self.products[:min(3, len(self.products))]
            
            # Generate many orders for same products
            for _ in range(duration_days * 8):
                product = random.choice(common_products)
                wo = self.generate_work_order(
                    product=product,
                    quantity=Decimal(random.randint(50, 150)),
                    due_date=datetime.now() + timedelta(days=random.randint(2, duration_days + 2)),
                    priority=random.choice([2, 3]),
                    status=WorkOrderStatus.PLANNED
                )
                work_orders.append(wo)
        
        return {
            'name': 'Bottleneck Test',
            'description': 'Concentrated orders on limited products',
            'work_orders': work_orders,
            'metrics': {
                'total_orders': len(work_orders),
                'unique_products': len(set(wo.product_id for wo in work_orders)),
                'orders_per_product': len(work_orders) / len(set(wo.product_id for wo in work_orders)) if work_orders else 0
            }
        }
    
    def save_work_orders(self, work_orders: List[WorkOrder], commit: bool = True) -> List[WorkOrder]:
        """
        Save work orders to database
        
        Args:
            work_orders: List of work orders to save
            commit: Whether to commit immediately
            
        Returns:
            List of saved work orders
        """
        for wo in work_orders:
            self.session.add(wo)
        
        if commit:
            self.session.commit()
            print(f"Saved {len(work_orders)} work orders to database")
        
        return work_orders
    
    def generate_for_simulation(self, 
                               simulation_hours: int = 8,
                               orders_per_hour: float = 2.0,
                               release_strategy: str = 'uniform') -> List[Dict[str, Any]]:
        """
        Generate work orders specifically for simulation
        
        Args:
            simulation_hours: Duration of simulation in hours
            orders_per_hour: Average orders to release per hour
            release_strategy: How to distribute releases ('uniform', 'front_loaded', 'back_loaded')
            
        Returns:
            List of dictionaries with work order and release time
        """
        simulation_orders = []
        total_orders = int(simulation_hours * orders_per_hour)
        
        for i in range(total_orders):
            # Calculate release time based on strategy
            if release_strategy == 'uniform':
                release_hour = i / orders_per_hour
            elif release_strategy == 'front_loaded':
                # 70% in first half
                if i < total_orders * 0.7:
                    release_hour = (i / (total_orders * 0.7)) * (simulation_hours * 0.5)
                else:
                    release_hour = simulation_hours * 0.5 + \
                                 ((i - total_orders * 0.7) / (total_orders * 0.3)) * (simulation_hours * 0.5)
            elif release_strategy == 'back_loaded':
                # 70% in second half
                if i < total_orders * 0.3:
                    release_hour = (i / (total_orders * 0.3)) * (simulation_hours * 0.5)
                else:
                    release_hour = simulation_hours * 0.5 + \
                                 ((i - total_orders * 0.3) / (total_orders * 0.7)) * (simulation_hours * 0.5)
            else:
                release_hour = random.uniform(0, simulation_hours)
            
            # Generate work order
            wo = self.generate_work_order(
                status=WorkOrderStatus.RELEASED,
                priority=random.choices([1, 2, 3, 4, 5], weights=[5, 15, 50, 20, 10])[0]
            )
            
            simulation_orders.append({
                'work_order': wo,
                'release_time': datetime.now() + timedelta(hours=release_hour),
                'lot_size': float(wo.order_quantity) / random.choice([1, 2, 3, 4])  # Split into lots
            })
        
        return simulation_orders
    
    def generate_report(self, work_orders: List[WorkOrder]) -> Dict[str, Any]:
        """
        Generate summary report for work orders
        
        Args:
            work_orders: List of work orders to analyze
            
        Returns:
            Dictionary with statistics and analysis
        """
        if not work_orders:
            return {'error': 'No work orders to analyze'}
        
        report = {
            'summary': {
                'total_orders': len(work_orders),
                'total_quantity': sum(float(wo.order_quantity) for wo in work_orders),
                'total_estimated_cost': sum(float(wo.estimated_cost or 0) for wo in work_orders),
                'date_range': {
                    'earliest_due': min(wo.due_date for wo in work_orders),
                    'latest_due': max(wo.due_date for wo in work_orders)
                }
            },
            'by_status': {},
            'by_priority': {},
            'by_product': {},
            'by_customer': {},
            'capacity_analysis': {}
        }
        
        # Group by status
        for wo in work_orders:
            status = wo.status.value if wo.status else 'unknown'
            if status not in report['by_status']:
                report['by_status'][status] = {'count': 0, 'quantity': 0}
            report['by_status'][status]['count'] += 1
            report['by_status'][status]['quantity'] += float(wo.order_quantity)
        
        # Group by priority
        for wo in work_orders:
            priority = f"Priority_{wo.priority}"
            if priority not in report['by_priority']:
                report['by_priority'][priority] = {'count': 0, 'quantity': 0}
            report['by_priority'][priority]['count'] += 1
            report['by_priority'][priority]['quantity'] += float(wo.order_quantity)
        
        # Group by product
        for wo in work_orders:
            if wo.product:
                product = wo.product.part_number
                if product not in report['by_product']:
                    report['by_product'][product] = {'count': 0, 'quantity': 0}
                report['by_product'][product]['count'] += 1
                report['by_product'][product]['quantity'] += float(wo.order_quantity)
        
        # Group by customer
        for wo in work_orders:
            customer = wo.customer_name or 'Unknown'
            if customer not in report['by_customer']:
                report['by_customer'][customer] = {'count': 0, 'quantity': 0}
            report['by_customer'][customer]['count'] += 1
            report['by_customer'][customer]['quantity'] += float(wo.order_quantity)
        
        # Capacity analysis
        total_hours = 0
        for wo in work_orders:
            if wo.route and wo.route.total_cycle_time:
                hours = float(wo.route.total_cycle_time * wo.order_quantity / 60)
                total_hours += hours
        
        report['capacity_analysis'] = {
            'total_production_hours': total_hours,
            'avg_hours_per_order': total_hours / len(work_orders) if work_orders else 0,
            'daily_capacity_required': total_hours / 7 if total_hours > 0 else 0  # Assume 7-day period
        }
        
        return report


# ============================================================================
# INTEGRATION WITH SIMULATOR
# ============================================================================

def generate_and_prepare_for_simulation(connection_string: str, 
                                       scenario_type: str = 'normal',
                                       duration_days: int = 7) -> Dict[str, Any]:
    """
    Generate work orders and prepare them for simulation
    
    Args:
        connection_string: Database connection string
        scenario_type: Type of scenario to generate
        duration_days: Duration of scenario
        
    Returns:
        Dictionary with generated work orders and simulation parameters
    """
    
    # Create session
    engine = create_engine(connection_string)
    session = get_session(engine)
    
    # Create generator
    generator = WorkOrderGenerator(session)
    
    # Generate scenario
    scenario = generator.generate_scenario(scenario_type, duration_days)
    
    # Save work orders
    saved_orders = generator.save_work_orders(scenario['work_orders'])
    
    # Prepare simulation data
    simulation_data = {
        'scenario': scenario,
        'work_orders': saved_orders,
        'simulation_parameters': {
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(days=duration_days),
            'release_schedule': []
        }
    }
    
    # Create release schedule for simulation
    for i, wo in enumerate(saved_orders):
        # Stagger releases over first day
        release_time = datetime.now() + timedelta(hours=i * 0.5)
        simulation_data['simulation_parameters']['release_schedule'].append({
            'work_order_id': wo.id,
            'release_time': release_time,
            'lot_size': float(wo.order_quantity) / 3  # Split into 3 lots
        })
    
    # Generate report
    report = generator.generate_report(saved_orders)
    simulation_data['generation_report'] = report
    
    session.close()
    
    return simulation_data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Database connection
    connection_string = 'postgresql://myuser:mypassword@localhost:5433/manufacturing'
    
    # Create session
    engine = create_engine(connection_string)
    session = get_session(engine)
    
    # Initialize generator
    generator = WorkOrderGenerator(session)
    
    print("="*60)
    print("WORK ORDER GENERATOR")
    print("="*60)
    
    # Generate single work order
    print("\n1. Generating single work order...")
    single_wo = generator.generate_work_order(
        quantity=Decimal(100),
        priority=2,
        notes="Test work order"
    )
    print(f"   Generated: {single_wo.work_order_number}")
    
    # Generate batch
    print("\n2. Generating batch of 5 work orders...")
    batch = generator.generate_batch(5)
    for wo in batch:
        print(f"   - {wo.work_order_number}: {wo.product.part_number if wo.product else 'N/A'} x {wo.order_quantity}")
    
    # Generate scenario
    print("\n3. Generating 'high_volume' scenario...")
    scenario = generator.generate_scenario('high_volume', duration_days=3)
    print(f"   Scenario: {scenario['name']}")
    print(f"   Orders generated: {scenario['metrics']['total_orders']}")
    print(f"   Total units: {scenario['metrics']['total_units']}")
    
    # Save to database
    print("\n4. Saving work orders to database...")
    saved = generator.save_work_orders(batch + [single_wo])
    
    # Generate for simulation
    print("\n5. Generating orders for simulation...")
    sim_orders = generator.generate_for_simulation(
        simulation_hours=8,
        orders_per_hour=2,
        release_strategy='uniform'
    )
    print(f"   Generated {len(sim_orders)} orders for simulation")
    
    # Generate report
    print("\n6. Generating report...")
    all_orders = batch + [single_wo] + scenario['work_orders']
    report = generator.generate_report(all_orders)
    
    print("\nREPORT SUMMARY:")
    print(f"   Total orders: {report['summary']['total_orders']}")
    print(f"   Total quantity: {report['summary']['total_quantity']:.0f}")
    print(f"   Total estimated cost: ${report['summary']['total_estimated_cost']:.2f}")
    print(f"   Required capacity: {report['capacity_analysis']['total_production_hours']:.1f} hours")
    
    print("\n   By Priority:")
    for priority, data in sorted(report['by_priority'].items()):
        print(f"      {priority}: {data['count']} orders, {data['quantity']:.0f} units")
    
    session.close()
    print("\nWork order generation complete!")