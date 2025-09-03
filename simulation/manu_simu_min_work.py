"""
Manufacturing Simulation Module - Minimal Working Version
Connects to existing database and simulates production flow
"""

from datetime import datetime, timedelta
import heapq
import random
import json
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import from your existing modules
from manufacturing.datamodel import (
    get_session, create_engine,
    WorkOrder, Operation, Equipment, WIPRecord, 
    WorkOrderStatus, WIPStatus, EquipmentStatus,
    Route, Sequence, Product
)

# ============================================================================
# EVENT SYSTEM
# ============================================================================

class EventType(Enum):
    """Types of simulation events"""
    LOT_RELEASE = "lot_release"
    OPERATION_START = "operation_start"
    OPERATION_COMPLETE = "operation_complete"
    EQUIPMENT_BREAKDOWN = "equipment_breakdown"
    EQUIPMENT_REPAIR = "equipment_repair"
    LOT_MOVE = "lot_move"
    SHIFT_CHANGE = "shift_change"

@dataclass(order=True)
class SimEvent:
    """Base simulation event"""
    timestamp: datetime
    event_type: EventType = field(compare=False)
    data: dict = field(default_factory=dict, compare=False)
    priority: int = field(default=5, compare=False)
    
    def __post_init__(self):
        # For heap sorting, combine timestamp and priority
        self.sort_key = (self.timestamp, -self.priority)

# ============================================================================
# SIMULATED ENTITIES
# ============================================================================

class SimulatedLot:
    """Represents a lot moving through the factory"""
    
    def __init__(self, lot_id: str, work_order: WorkOrder, quantity: float, release_time: datetime):
        self.lot_id = lot_id
        self.work_order = work_order
        self.quantity = quantity
        self.release_time = release_time
        self.current_operation_index = 0
        self.operation_history = []
        self.current_status = WIPStatus.QUEUE
        self.completed = False
        
        # Get all operations for this work order's route
        self.operations = self._get_operations_list()
        
    def _get_operations_list(self):
        """Extract ordered list of operations from route"""
        operations = []
        if self.work_order.route:
            for sequence in sorted(self.work_order.route.sequences, 
                                  key=lambda s: s.sequence_number):
                for operation in sorted(sequence.operations, 
                                       key=lambda o: o.operation_number):
                    operations.append(operation)
        return operations
    
    def get_current_operation(self) -> Optional[Operation]:
        """Get current operation to be performed"""
        if self.current_operation_index < len(self.operations):
            return self.operations[self.current_operation_index]
        return None
    
    def complete_current_operation(self, completion_time: datetime, good_qty: float, scrap_qty: float = 0):
        """Record completion of current operation"""
        if self.get_current_operation():
            self.operation_history.append({
                'operation': self.get_current_operation(),
                'start_time': completion_time - timedelta(minutes=float(self.get_current_operation().cycle_time_minutes or 60)),
                'end_time': completion_time,
                'good_quantity': good_qty,
                'scrap_quantity': scrap_qty
            })
            self.current_operation_index += 1
            
            # Update quantity
            self.quantity = good_qty
            
            # Check if lot is completed
            if self.current_operation_index >= len(self.operations):
                self.completed = True
                self.current_status = WIPStatus.COMPLETED

class SimulatedEquipment:
    """Represents equipment in simulation"""
    
    def __init__(self, equipment: Equipment):
        self.equipment = equipment
        self.busy_until = None
        self.current_lot = None
        self.utilization_log = []
        self.breakdown_schedule = []
        self.total_busy_time = 0
        self.total_available_time = 0
        
    def is_available(self, timestamp: datetime) -> bool:
        """Check if equipment is available at given time"""
        if self.busy_until and timestamp < self.busy_until:
            return False
            
        # Check breakdown schedule
        for breakdown_start, breakdown_end in self.breakdown_schedule:
            if breakdown_start <= timestamp <= breakdown_end:
                return False
                
        return True
    
    def allocate(self, lot: SimulatedLot, operation: Operation, start_time: datetime):
        """Allocate equipment to a lot"""
        cycle_time = float(operation.cycle_time_minutes or 60)
        setup_time = float(operation.setup_time_minutes or 0)
        total_time = cycle_time + setup_time
        
        self.busy_until = start_time + timedelta(minutes=total_time)
        self.current_lot = lot
        
        # Log utilization
        self.utilization_log.append({
            'lot_id': lot.lot_id,
            'operation': operation.operation_code,
            'start': start_time,
            'end': self.busy_until,
            'duration': total_time
        })
        
        self.total_busy_time += total_time
        
    def release(self):
        """Release equipment after operation completion"""
        self.current_lot = None

# ============================================================================
# RESOURCE MANAGER
# ============================================================================

class ResourceManager:
    """Manages equipment allocation"""
    
    def __init__(self, session):
        self.session = session
        self.equipment_pool = {}
        self._load_equipment()
        
    def _load_equipment(self):
        """Load equipment from database"""
        equipment_list = self.session.query(Equipment).filter(
            Equipment.status.in_([EquipmentStatus.AVAILABLE, EquipmentStatus.IN_USE])
        ).all()
        
        for eq in equipment_list:
            self.equipment_pool[eq.id] = SimulatedEquipment(eq)
    
    def find_available_equipment(self, operation: Operation, timestamp: datetime) -> Optional[SimulatedEquipment]:
        """Find available equipment for an operation"""
        # Get equipment that can perform this operation
        suitable_equipment = []
        for eq in operation.equipment:
            if eq.id in self.equipment_pool:
                sim_eq = self.equipment_pool[eq.id]
                if sim_eq.is_available(timestamp):
                    suitable_equipment.append(sim_eq)
        
        # Return first available (can implement better selection logic)
        if suitable_equipment:
            return suitable_equipment[0]
        return None
    
    def allocate_equipment(self, equipment: SimulatedEquipment, lot: SimulatedLot, 
                         operation: Operation, start_time: datetime):
        """Allocate equipment to lot"""
        equipment.allocate(lot, operation, start_time)
        
    def calculate_utilization(self, simulation_duration_hours: float) -> dict:
        """Calculate equipment utilization statistics"""
        utilization_stats = {}
        
        for eq_id, sim_eq in self.equipment_pool.items():
            if simulation_duration_hours > 0:
                utilization = (sim_eq.total_busy_time / 60) / simulation_duration_hours * 100
            else:
                utilization = 0
                
            utilization_stats[sim_eq.equipment.code] = {
                'utilization_percent': min(100, utilization),
                'total_busy_hours': sim_eq.total_busy_time / 60,
                'number_of_lots': len(sim_eq.utilization_log)
            }
            
        return utilization_stats

# ============================================================================
# STATISTICS COLLECTOR
# ============================================================================

class StatisticsCollector:
    """Collects simulation statistics"""
    
    def __init__(self):
        self.events_log = []
        self.lots_released = 0
        self.lots_completed = 0
        self.total_cycle_time = 0
        self.operation_metrics = {}
        self.bottleneck_queues = {}
        
    def record_event(self, event: SimEvent):
        """Record an event"""
        self.events_log.append({
            'timestamp': event.timestamp.isoformat(),
            'type': event.event_type.value,
            'data': event.data
        })
        
        # Update specific metrics
        if event.event_type == EventType.LOT_RELEASE:
            self.lots_released += 1
        elif event.event_type == EventType.OPERATION_COMPLETE:
            self._record_operation_completion(event)
            
    def _record_operation_completion(self, event: SimEvent):
        """Record operation completion metrics"""
        operation_code = event.data.get('operation_code')
        cycle_time = event.data.get('cycle_time', 0)
        
        if operation_code not in self.operation_metrics:
            self.operation_metrics[operation_code] = {
                'completions': 0,
                'total_cycle_time': 0,
                'avg_cycle_time': 0,
                'min_cycle_time': float('inf'),
                'max_cycle_time': 0
            }
        
        metrics = self.operation_metrics[operation_code]
        metrics['completions'] += 1
        metrics['total_cycle_time'] += cycle_time
        metrics['avg_cycle_time'] = metrics['total_cycle_time'] / metrics['completions']
        metrics['min_cycle_time'] = min(metrics['min_cycle_time'], cycle_time)
        metrics['max_cycle_time'] = max(metrics['max_cycle_time'], cycle_time)
        
    def record_lot_completion(self, lot: SimulatedLot, completion_time: datetime):
        """Record lot completion"""
        self.lots_completed += 1
        total_time = (completion_time - lot.release_time).total_seconds() / 3600  # hours
        self.total_cycle_time += total_time
        
    def get_summary(self) -> dict:
        """Get simulation summary statistics"""
        avg_cycle_time = self.total_cycle_time / self.lots_completed if self.lots_completed > 0 else 0
        
        return {
            'lots_released': self.lots_released,
            'lots_completed': self.lots_completed,
            'completion_rate': (self.lots_completed / self.lots_released * 100) if self.lots_released > 0 else 0,
            'avg_cycle_time_hours': avg_cycle_time,
            'operation_metrics': self.operation_metrics,
            'total_events': len(self.events_log)
        }

# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================

class ManufacturingSimulator:
    """Main simulation engine"""
    
    def __init__(self, session, start_time: datetime, end_time: datetime, 
                 random_seed: int = 42):
        self.session = session
        self.current_time = start_time
        self.start_time = start_time
        self.end_time = end_time
        self.event_queue = []
        self.active_lots = {}
        self.completed_lots = []
        self.resource_manager = ResourceManager(session)
        self.statistics = StatisticsCollector()
        self.operation_queues = {}  # operation_id -> list of waiting lots
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Lot ID counter
        self.lot_counter = 0
        
    def schedule_event(self, event: SimEvent):
        """Add event to queue"""
        heapq.heappush(self.event_queue, event)
        
    def schedule_work_order(self, work_order: WorkOrder, release_time: datetime, 
                          lot_size: Optional[float] = None):
        """Schedule a work order for production"""
        if lot_size is None:
            lot_size = float(work_order.order_quantity)
            
        # Create lot release event
        event = SimEvent(
            timestamp=release_time,
            event_type=EventType.LOT_RELEASE,
            data={
                'work_order_id': work_order.id,
                'work_order_number': work_order.work_order_number,
                'quantity': lot_size,
                'product': work_order.product.part_number if work_order.product else 'Unknown'
            },
            priority=1
        )
        self.schedule_event(event)
        
    def run(self):
        """Run simulation"""
        print(f"Starting simulation from {self.start_time} to {self.end_time}")
        
        events_processed = 0
        while self.event_queue and self.current_time <= self.end_time:
            event = heapq.heappop(self.event_queue)
            
            # Don't process events beyond end time
            if event.timestamp > self.end_time:
                break
                
            self.current_time = event.timestamp
            self.process_event(event)
            self.statistics.record_event(event)
            
            events_processed += 1
            if events_processed % 100 == 0:
                print(f"Processed {events_processed} events, current time: {self.current_time}")
        
        # Calculate final statistics
        simulation_duration = (self.end_time - self.start_time).total_seconds() / 3600
        self.equipment_utilization = self.resource_manager.calculate_utilization(simulation_duration)
        
        print(f"Simulation complete. Processed {events_processed} events.")
        
    def process_event(self, event: SimEvent):
        """Process a simulation event"""
        if event.event_type == EventType.LOT_RELEASE:
            self._process_lot_release(event)
        elif event.event_type == EventType.OPERATION_START:
            self._process_operation_start(event)
        elif event.event_type == EventType.OPERATION_COMPLETE:
            self._process_operation_complete(event)
        elif event.event_type == EventType.LOT_MOVE:
            self._process_lot_move(event)
            
    def _process_lot_release(self, event: SimEvent):
        """Process lot release event"""
        # Get work order
        work_order = self.session.query(WorkOrder).get(event.data['work_order_id'])
        if not work_order:
            return
            
        # Create simulated lot
        self.lot_counter += 1
        lot_id = f"SIM-LOT-{self.lot_counter:04d}"
        lot = SimulatedLot(lot_id, work_order, event.data['quantity'], event.timestamp)
        
        self.active_lots[lot_id] = lot
        
        # Schedule first operation
        first_operation = lot.get_current_operation()
        if first_operation:
            # Add some delay for lot preparation
            prep_time = random.uniform(5, 15)  # minutes
            self.schedule_event(SimEvent(
                timestamp=event.timestamp + timedelta(minutes=prep_time),
                event_type=EventType.OPERATION_START,
                data={'lot_id': lot_id},
                priority=2
            ))
            
    def _process_operation_start(self, event: SimEvent):
        """Process operation start event"""
        lot_id = event.data['lot_id']
        lot = self.active_lots.get(lot_id)
        
        if not lot or lot.completed:
            return
            
        operation = lot.get_current_operation()
        if not operation:
            return
            
        # Find available equipment
        equipment = self.resource_manager.find_available_equipment(operation, event.timestamp)
        
        if equipment:
            # Allocate equipment and start operation
            self.resource_manager.allocate_equipment(equipment, lot, operation, event.timestamp)
            lot.current_status = WIPStatus.IN_PROCESS
            
            # Calculate processing time with variability
            base_cycle_time = float(operation.cycle_time_minutes or 60)
            setup_time = float(operation.setup_time_minutes or 0)
            
            # Add random variation (±20%)
            cycle_time = base_cycle_time * random.uniform(0.8, 1.2)
            total_time = cycle_time + setup_time
            
            # Schedule completion
            self.schedule_event(SimEvent(
                timestamp=event.timestamp + timedelta(minutes=total_time),
                event_type=EventType.OPERATION_COMPLETE,
                data={
                    'lot_id': lot_id,
                    'operation_code': operation.operation_code,
                    'equipment_id': equipment.equipment.id,
                    'cycle_time': cycle_time
                },
                priority=3
            ))
        else:
            # No equipment available, queue the lot
            if operation.id not in self.operation_queues:
                self.operation_queues[operation.id] = []
            
            if lot_id not in [l['lot_id'] for l in self.operation_queues[operation.id]]:
                self.operation_queues[operation.id].append({
                    'lot_id': lot_id,
                    'queued_at': event.timestamp
                })
                lot.current_status = WIPStatus.QUEUE
                
            # Retry after some time
            retry_time = random.uniform(10, 30)  # minutes
            self.schedule_event(SimEvent(
                timestamp=event.timestamp + timedelta(minutes=retry_time),
                event_type=EventType.OPERATION_START,
                data={'lot_id': lot_id},
                priority=2
            ))
            
    def _process_operation_complete(self, event: SimEvent):
        """Process operation completion event"""
        lot_id = event.data['lot_id']
        lot = self.active_lots.get(lot_id)
        
        if not lot:
            return
            
        # Calculate yield (simulate quality)
        base_quantity = lot.quantity
        scrap_rate = random.uniform(0, 0.05)  # 0-5% scrap
        good_quantity = base_quantity * (1 - scrap_rate)
        scrap_quantity = base_quantity * scrap_rate
        
        # Complete current operation
        lot.complete_current_operation(event.timestamp, good_quantity, scrap_quantity)
        
        # Release equipment
        equipment_id = event.data['equipment_id']
        if equipment_id in self.resource_manager.equipment_pool:
            equipment = self.resource_manager.equipment_pool[equipment_id]
            equipment.release()
            
            # Check if there are queued lots for this equipment
            # (simplified - should check which operations the equipment can handle)
            for op_id, queue in self.operation_queues.items():
                if queue:
                    # Process next queued lot
                    next_lot_data = queue.pop(0)
                    self.schedule_event(SimEvent(
                        timestamp=event.timestamp + timedelta(minutes=1),
                        event_type=EventType.OPERATION_START,
                        data={'lot_id': next_lot_data['lot_id']},
                        priority=1
                    ))
                    break
        
        # Check if lot is completed
        if lot.completed:
            self.completed_lots.append(lot)
            del self.active_lots[lot_id]
            self.statistics.record_lot_completion(lot, event.timestamp)
            print(f"Lot {lot_id} completed at {event.timestamp}")
        else:
            # Schedule move to next operation
            move_time = random.uniform(5, 20)  # minutes
            self.schedule_event(SimEvent(
                timestamp=event.timestamp + timedelta(minutes=move_time),
                event_type=EventType.LOT_MOVE,
                data={'lot_id': lot_id},
                priority=2
            ))
            
    def _process_lot_move(self, event: SimEvent):
        """Process lot movement between operations"""
        lot_id = event.data['lot_id']
        lot = self.active_lots.get(lot_id)
        
        if not lot or lot.completed:
            return
            
        # Update status
        lot.current_status = WIPStatus.QUEUE
        
        # Schedule next operation start
        self.schedule_event(SimEvent(
            timestamp=event.timestamp + timedelta(minutes=1),
            event_type=EventType.OPERATION_START,
            data={'lot_id': lot_id},
            priority=2
        ))
        
    def get_results(self) -> dict:
        """Get simulation results"""
        results = {
            'summary': self.statistics.get_summary(),
            'equipment_utilization': self.equipment_utilization,
            'completed_lots': [
                {
                    'lot_id': lot.lot_id,
                    'work_order': lot.work_order.work_order_number,
                    'quantity_in': lot.operation_history[0]['good_quantity'] if lot.operation_history else lot.quantity,
                    'quantity_out': lot.quantity,
                    'total_time_hours': (lot.operation_history[-1]['end_time'] - lot.release_time).total_seconds() / 3600 if lot.operation_history else 0,
                    'operations_completed': len(lot.operation_history)
                }
                for lot in self.completed_lots
            ],
            'active_lots': [
                {
                    'lot_id': lot.lot_id,
                    'current_operation': lot.get_current_operation().operation_code if lot.get_current_operation() else 'None',
                    'status': lot.current_status.value
                }
                for lot in self.active_lots.values()
            ]
        }
        
        return results

# ============================================================================
# SCENARIO MANAGER
# ============================================================================

class SimulationScenario:
    """Defines a simulation scenario"""
    
    def __init__(self, name: str, start_time: datetime, end_time: datetime):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.work_orders = []
        self.equipment_failures = []
        self.parameters = {}
        
    def add_work_order(self, work_order: WorkOrder, release_time: datetime, lot_size: Optional[float] = None):
        """Add work order to scenario"""
        self.work_orders.append({
            'work_order': work_order,
            'release_time': release_time,
            'lot_size': lot_size
        })
        
    def add_equipment_failure(self, equipment_id: int, failure_start: datetime, repair_duration_hours: float):
        """Add equipment failure to scenario"""
        self.equipment_failures.append({
            'equipment_id': equipment_id,
            'start': failure_start,
            'end': failure_start + timedelta(hours=repair_duration_hours)
        })
        
    def set_parameter(self, key: str, value):
        """Set scenario parameter"""
        self.parameters[key] = value

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_simulation(connection_string: str, scenario: SimulationScenario) -> dict:
    """Run a simulation scenario"""
    
    # Create database connection
    engine = create_engine(connection_string)
    session = get_session(engine)
    
    # Create simulator
    simulator = ManufacturingSimulator(
        session=session,
        start_time=scenario.start_time,
        end_time=scenario.end_time,
        random_seed=scenario.parameters.get('random_seed', 42)
    )
    
    # Schedule work orders
    for wo_data in scenario.work_orders:
        simulator.schedule_work_order(
            work_order=wo_data['work_order'],
            release_time=wo_data['release_time'],
            lot_size=wo_data['lot_size']
        )
    
    # Add equipment failures
    for failure in scenario.equipment_failures:
        if failure['equipment_id'] in simulator.resource_manager.equipment_pool:
            equipment = simulator.resource_manager.equipment_pool[failure['equipment_id']]
            equipment.breakdown_schedule.append((failure['start'], failure['end']))
    
    # Run simulation
    simulator.run()
    
    # Get results
    results = simulator.get_results()
    results['scenario_name'] = scenario.name
    
    # Close session
    session.close()
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Connection string
    connection_string = 'postgresql://myuser:mypassword@localhost:5433/manufacturing'
    
    # Create session for loading work orders
    engine = create_engine(connection_string)
    session = get_session(engine)
    
    # Create scenario
    scenario = SimulationScenario(
        name="Base Production Plan",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=8)
    )
    
    # Load planned work orders
    work_orders = session.query(WorkOrder).filter_by(
        status=WorkOrderStatus.PLANNED
    ).limit(3).all()
    
    # Schedule work orders
    for i, wo in enumerate(work_orders):
        release_time = scenario.start_time + timedelta(minutes=i * 30)
        # Split into 3 lots
        lot_size = float(wo.order_quantity) / 3
        scenario.add_work_order(wo, release_time, lot_size)
    
    session.close()
    
    # Run simulation
    print("Running simulation...")
    results = run_simulation(connection_string, scenario)
    
    # Print results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Scenario: {results['scenario_name']}")
    print(f"Lots Released: {results['summary']['lots_released']}")
    print(f"Lots Completed: {results['summary']['lots_completed']}")
    print(f"Completion Rate: {results['summary']['completion_rate']:.1f}%")
    print(f"Avg Cycle Time: {results['summary']['avg_cycle_time_hours']:.2f} hours")
    
    print("\nEquipment Utilization:")
    for eq_code, stats in results['equipment_utilization'].items():
        print(f"  {eq_code}: {stats['utilization_percent']:.1f}% ({stats['number_of_lots']} lots)")
    
    print("\nOperation Metrics:")
    for op_code, metrics in results['summary']['operation_metrics'].items():
        print(f"  {op_code}: {metrics['completions']} completions, avg time: {metrics['avg_cycle_time']:.1f} min")