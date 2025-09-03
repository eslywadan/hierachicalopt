"""
WIP History Table for Real-Time Manufacturing Tracking
Extends the existing manufacturing.py model with comprehensive history tracking
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal
from enum import Enum
from typing import Optional, List
import json
import random
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, DECIMAL, JSON, Index, event, select, func, and_
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

# Import existing enums from manufacturing.py
from manufacturing.datamodel import Base, WIPStatus, WIPRecord

local_tz = ZoneInfo("Asia/Taipei")

# ============================================================================
# WIP HISTORY TRACKING
# ============================================================================

class WIPEventType(Enum):
    """Types of events that can occur in WIP tracking"""
    CREATED = "created"                    # Lot created
    QUEUE_ENTRY = "queue_entry"            # Entered queue at operation
    SETUP_START = "setup_start"            # Setup started
    SETUP_COMPLETE = "setup_complete"      # Setup completed
    PROCESS_START = "process_start"        # Processing started
    PROCESS_PAUSE = "process_pause"        # Processing paused
    PROCESS_RESUME = "process_resume"      # Processing resumed
    PROCESS_COMPLETE = "process_complete"  # Processing completed
    INSPECTION_START = "inspection_start"  # Quality inspection started
    INSPECTION_PASS = "inspection_pass"    # Passed inspection
    INSPECTION_FAIL = "inspection_fail"    # Failed inspection
    REWORK_START = "rework_start"         # Rework started
    REWORK_COMPLETE = "rework_complete"   # Rework completed
    SCRAP = "scrap"                       # Material scrapped
    MOVE_START = "move_start"             # Moving to next operation
    MOVE_COMPLETE = "move_complete"       # Arrived at next operation
    HOLD = "hold"                          # Put on hold
    RELEASE = "release"                   # Released from hold
    COMPLETE = "complete"                  # Lot completed
    CANCELLED = "cancelled"                # Lot cancelled

class WIPHistory(Base):
    """Track every event and movement of WIP through the factory"""
    __tablename__ = 'wip_history'
    
    id = Column(Integer, primary_key=True)
    
    # WIP Reference
    wip_record_id = Column(Integer, ForeignKey('wip_records.id', ondelete='CASCADE'))
    wip_lot_number = Column(String(50), nullable=False, index=True)
    work_order_number = Column(String(50), nullable=False, index=True)
    
    # Event Information
    event_type = Column(String(30), nullable=False, index=True)
    event_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Location at time of event
    operation_id = Column(Integer, ForeignKey('operations.id'))
    operation_code = Column(String(50))  # Denormalized for history preservation
    equipment_id = Column(Integer, ForeignKey('equipment.id'))
    equipment_code = Column(String(50))  # Denormalized for history preservation
    plant = Column(String(50))
    work_center = Column(String(50))
    
    # Quantities at time of event
    quantity = Column(DECIMAL(10, 2))
    good_quantity = Column(DECIMAL(10, 2))
    scrap_quantity = Column(DECIMAL(10, 2))
    rework_quantity = Column(DECIMAL(10, 2))
    
    # Status transitions
    previous_status = Column(String(20))
    new_status = Column(String(20))
    
    # Performance metrics captured at event
    cycle_time_minutes = Column(DECIMAL(10, 2))
    setup_time_minutes = Column(DECIMAL(10, 2))
    queue_time_minutes = Column(DECIMAL(10, 2))
    move_time_minutes = Column(DECIMAL(10, 2))
    
    # Resource information
    operator_id = Column(String(50))
    operator_name = Column(String(100))
    shift = Column(String(20))
    
    # Quality data
    quality_check_result = Column(String(20))  # pass/fail/na
    defect_codes = Column(JSON)  # Array of defect codes if inspection failed
    measurement_data = Column(JSON)  # Any measurements taken
    
    # Additional context
    notes = Column(Text)
    reason_code = Column(String(50))  # Reason for hold, scrap, etc.
    
    # System tracking
    created_by = Column(String(50))
    terminal_id = Column(String(50))  # Which terminal/scanner logged this
    
    # Relationships
    wip_record = relationship('WIPRecord', backref='history')
    operation = relationship('Operation')
    equipment = relationship('Equipment')
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_history_timestamp', 'event_timestamp'),
        Index('idx_history_lot', 'wip_lot_number', 'event_timestamp'),
        Index('idx_history_operation', 'operation_id', 'event_timestamp'),
        Index('idx_history_equipment', 'equipment_id', 'event_timestamp'),
        Index('idx_history_work_order', 'work_order_number', 'event_timestamp'),
    )
    
    def __repr__(self):
        return f"<WIPHistory {self.wip_lot_number} {self.event_type} at {self.event_timestamp}>"


class WIPSnapshot(Base):
    """Periodic snapshots of WIP status for reporting and analytics"""
    __tablename__ = 'wip_snapshots'
    
    id = Column(Integer, primary_key=True)
    snapshot_timestamp = Column(DateTime, nullable=False, index=True)
    
    # WIP identification
    wip_record_id = Column(Integer, ForeignKey('wip_records.id'))
    wip_lot_number = Column(String(50), nullable=False)
    work_order_number = Column(String(50), nullable=False)
    product_part_number = Column(String(50))
    
    # Current state at snapshot time
    operation_code = Column(String(50))
    equipment_code = Column(String(50))
    status = Column(String(20))
    
    # Quantities
    quantity = Column(DECIMAL(10, 2))
    good_quantity = Column(DECIMAL(10, 2))
    scrap_quantity = Column(DECIMAL(10, 2))
    
    # Time in current state
    time_in_status_minutes = Column(DECIMAL(10, 2))
    total_cycle_time_minutes = Column(DECIMAL(10, 2))
    
    # Value calculation
    material_value = Column(DECIMAL(12, 2))
    labor_value = Column(DECIMAL(12, 2))
    overhead_value = Column(DECIMAL(12, 2))
    total_value = Column(DECIMAL(12, 2))
    
    # Indexes
    __table_args__ = (
        Index('idx_snapshot_timestamp', 'snapshot_timestamp'),
        Index('idx_snapshot_lot', 'wip_lot_number', 'snapshot_timestamp'),
    )


# ============================================================================
# AUTOMATIC HISTORY TRACKING
# ============================================================================

def create_wip_history_entry(wip_record, event_type, **kwargs):
    """Create a history entry for a WIP event"""
    history = WIPHistory(
        wip_record_id=wip_record.id,
        wip_lot_number=wip_record.wip_lot_number,
        work_order_number=wip_record.work_order.work_order_number if wip_record.work_order else None,
        event_type=event_type,
        event_timestamp=kwargs.get('timestamp', datetime.now(local_tz)),
        
        # Current location
        operation_id=wip_record.current_operation_id,
        operation_code=wip_record.current_operation.operation_code if wip_record.current_operation else None,
        equipment_id=wip_record.current_equipment_id,
        equipment_code=wip_record.equipment.code if wip_record.equipment else None,
        plant=wip_record.plant,
        work_center=wip_record.work_center,
        
        # Quantities
        quantity=wip_record.quantity,
        good_quantity=wip_record.good_quantity,
        scrap_quantity=wip_record.scrap_quantity,
        rework_quantity=wip_record.rework_quantity,
        
        # Status
        previous_status=kwargs.get('previous_status'),
        new_status=wip_record.status.value if wip_record.status else None,
        
        # Additional data from kwargs
        operator_id=kwargs.get('operator_id'),
        operator_name=kwargs.get('operator_name'),
        shift=kwargs.get('shift'),
        notes=kwargs.get('notes'),
        reason_code=kwargs.get('reason_code'),
        quality_check_result=kwargs.get('quality_check_result'),
        defect_codes=kwargs.get('defect_codes'),
        measurement_data=kwargs.get('measurement_data'),
        created_by=kwargs.get('created_by', 'system'),
        terminal_id=kwargs.get('terminal_id')
    )
    
    return history


# SQLAlchemy event listeners for automatic history tracking
@event.listens_for(WIPRecord, 'after_insert')
def log_wip_creation(mapper, connection, target):
    """Automatically log when a WIP record is created"""
    history = WIPHistory(
        wip_record_id=target.id,
        wip_lot_number=target.wip_lot_number,
        work_order_number=target.work_order.work_order_number if target.work_order else None,
        event_type=WIPEventType.CREATED.value,
        event_timestamp=datetime.now(local_tz),
        quantity=target.quantity,
        new_status=target.status.value if target.status else None,
        created_by='system'
    )
    # connection.execute(WIPHistory.__table__.insert().values(**history.__dict__))
    connection.execute(
        WIPHistory.__table__.insert().values(
        {col.name: getattr(history, col.name) for col in WIPHistory.__table__.columns if col.name != "id"}
    )
)


# ============================================================================
# REAL-TIME TRACKING FUNCTIONS
# ============================================================================

class WIPTracker:
    """Real-time WIP tracking operations"""
    
    @staticmethod
    def start_operation(session, wip_lot_number, operator_id, terminal_id=None):
        """Start processing a WIP lot at current operation"""
        wip = session.query(WIPRecord).filter_by(wip_lot_number=wip_lot_number).first()
        if not wip:
            return False
        
        # Update WIP record
        previous_status = wip.status.value if wip.status else None
        wip.status = WIPStatus.IN_PROCESS
        wip.process_start_time = datetime.now(local_tz)
        
        # Create history entry
        history = create_wip_history_entry(
            wip, 
            WIPEventType.PROCESS_START.value,
            previous_status=previous_status,
            operator_id=operator_id,
            terminal_id=terminal_id,
            timestamp=wip.process_start_time
        )
        session.add(history)
        session.commit()
        
        return True
    
    @staticmethod
    def complete_operation(session, wip_lot_number, good_qty, scrap_qty=0, 
                         defect_codes=None, operator_id=None):
        """Complete processing at current operation"""
        wip = session.query(WIPRecord).filter_by(wip_lot_number=wip_lot_number).first()
        if not wip:
            return False
        
        # Validate before saving
        errors = WIPValidator.validate_quantities(
        wip.quantity, good_qty, scrap_qty
        )
    
        if errors:
            print(f"Validation errors: {errors}")
            return False

        # Calculate cycle time
        process_end = datetime.now(local_tz)
        cycle_time = None
        if wip.process_start_time:
            cycle_time = (process_end - wip.process_start_time).total_seconds() / 60
        
        # Update WIP record
        previous_status = wip.status.value if wip.status else None
        wip.status = WIPStatus.COMPLETED
        wip.process_end_time = process_end
        wip.good_quantity = good_qty
        wip.scrap_quantity = (wip.scrap_quantity or 0) + scrap_qty
        
        # Create history entry
        history = create_wip_history_entry(
            wip,
            WIPEventType.PROCESS_COMPLETE.value,
            previous_status=previous_status,
            cycle_time_minutes=cycle_time,
            defect_codes=defect_codes,
            operator_id=operator_id,
            timestamp=process_end
        )
        session.add(history)
        session.commit()
        
        return True
    
    @staticmethod
    def move_to_next_operation(session, wip_lot_number, next_operation_id, 
                              move_time_minutes=None):
        """Move WIP to next operation"""
        wip = session.query(WIPRecord).filter_by(wip_lot_number=wip_lot_number).first()
        if not wip:
            return False
        
        # Log move start
        move_start_history = create_wip_history_entry(
            wip,
            WIPEventType.MOVE_START.value,
            timestamp=datetime.now(local_tz)
        )
        session.add(move_start_history)
        
        # Update WIP location
        wip.current_operation_id = next_operation_id
        wip.status = WIPStatus.QUEUE
        wip.queue_entry_time = datetime.now(local_tz) + timedelta(minutes=move_time_minutes or 0)
        
        # Log move complete
        move_complete_history = create_wip_history_entry(
            wip,
            WIPEventType.MOVE_COMPLETE.value,
            move_time_minutes=move_time_minutes,
            timestamp=wip.queue_entry_time
        )
        session.add(move_complete_history)
        session.commit()
        
        return True
    
    @staticmethod
    def record_quality_check(session, wip_lot_number, passed, measurements=None, 
                           defect_codes=None, inspector_id=None):
        """Record quality inspection results"""
        wip = session.query(WIPRecord).filter_by(wip_lot_number=wip_lot_number).first()
        if not wip:
            return False
        
        event_type = WIPEventType.INSPECTION_PASS if passed else WIPEventType.INSPECTION_FAIL
        
        history = create_wip_history_entry(
            wip,
            event_type.value,
            quality_check_result='pass' if passed else 'fail',
            measurement_data=measurements,
            defect_codes=defect_codes,
            operator_id=inspector_id,
            timestamp=datetime.now(local_tz)
        )
        
        if not passed:
            wip.status = WIPStatus.REWORK
            wip.rework_quantity = (wip.rework_quantity or 0) + wip.quantity
        
        session.add(history)
        session.commit()
        
        return True
    
    @staticmethod
    def put_on_hold(session, wip_lot_number, reason_code, notes=None):
        """Put WIP lot on hold"""
        wip = session.query(WIPRecord).filter_by(wip_lot_number=wip_lot_number).first()
        if not wip:
            return False
        
        previous_status = wip.status.value if wip.status else None
        wip.status = WIPStatus.ON_HOLD
        
        history = create_wip_history_entry(
            wip,
            WIPEventType.HOLD.value,
            previous_status=previous_status,
            reason_code=reason_code,
            notes=notes,
            timestamp=datetime.now(local_tz)
        )
        
        session.add(history)
        session.commit()
        
        return True
    

class WIPValidator:
    """Validate WIP data before insertion"""
    
    @staticmethod
    def validate_quantities(quantity, good_qty, scrap_qty=0, rework_qty=0):
        """Validate quantity relationships"""
        errors = []
        
        if good_qty > quantity:
            errors.append(f"Good quantity ({good_qty}) exceeds total quantity ({quantity})")
        
        total_output = (good_qty or 0) + (scrap_qty or 0) + (rework_qty or 0)
        if total_output > quantity:
            errors.append(f"Total output ({total_output}) exceeds input quantity ({quantity})")
        
        yield_pct = (good_qty / quantity * 100) if quantity > 0 else 0
        if yield_pct > 99.5:  # Flag suspiciously high yields
            errors.append(f"Yield of {yield_pct:.1f}% seems unrealistically high")
        
        return errors



# ============================================================================
# ANALYTICS AND REPORTING
# ============================================================================

class WIPAnalytics:
    """Analytics functions for WIP history data"""
    
    @staticmethod
    def get_lot_timeline(session, wip_lot_number):
        """Get complete timeline of events for a lot"""
        history = session.query(WIPHistory).filter_by(
            wip_lot_number=wip_lot_number
        ).order_by(WIPHistory.event_timestamp).all()
        
        timeline = []
        for event in history:
            timeline.append({
                'timestamp': event.event_timestamp,
                'event': event.event_type,
                'operation': event.operation_code,
                'equipment': event.equipment_code,
                'status': event.new_status,
                'quantity': float(event.quantity) if event.quantity else None,
                'good_quantity': float(event.good_quantity) if event.good_quantity else None,
                'operator': event.operator_name or event.operator_id,
                'notes': event.notes
            })
        
        return timeline
    
    @staticmethod
    def calculate_operation_metrics(session, operation_id, start_date, end_date):
        """Calculate performance metrics for an operation"""
        
        # Get all completed operations in date range
        completed_events = session.query(WIPHistory).filter(
            WIPHistory.operation_id == operation_id,
            WIPHistory.event_type == WIPEventType.PROCESS_COMPLETE.value,
            WIPHistory.event_timestamp.between(start_date, end_date)
        ).all()
        
        if not completed_events:
            return None
        
        cycle_times = [float(e.cycle_time_minutes) for e in completed_events 
                      if e.cycle_time_minutes]
        quantities = [float(e.good_quantity or 0) for e in completed_events]
        scrap_quantities = [float(e.scrap_quantity or 0) for e in completed_events]
        
        metrics = {
            'total_lots_processed': len(completed_events),
            'total_quantity_produced': sum(quantities),
            'total_scrap': sum(scrap_quantities),
            'average_cycle_time': sum(cycle_times) / len(cycle_times) if cycle_times else 0,
            'min_cycle_time': min(cycle_times) if cycle_times else 0,
            'max_cycle_time': max(cycle_times) if cycle_times else 0,
            'first_pass_yield': (sum(quantities) / (sum(quantities) + sum(scrap_quantities)) * 100) 
                               if (sum(quantities) + sum(scrap_quantities)) > 0 else 0,
            'throughput_per_hour': sum(quantities) / (sum(cycle_times) / 60) if sum(cycle_times) > 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def get_current_wip_status(session):
        """Get real-time snapshot of all WIP in the factory"""
        
        # Get latest status for each WIP lot
        subquery = session.query(
            WIPHistory.wip_lot_number,
            func.max(WIPHistory.event_timestamp).label('latest_timestamp')
        ).group_by(WIPHistory.wip_lot_number).subquery()
        
        current_status = session.query(WIPHistory).join(
            subquery,
            and_(
                WIPHistory.wip_lot_number == subquery.c.wip_lot_number,
                WIPHistory.event_timestamp == subquery.c.latest_timestamp
            )
        ).all()
        
        wip_summary = {
            'total_lots': len(current_status),
            'by_status': {},
            'by_operation': {},
            'by_work_center': {}
        }
        
        for record in current_status:
            # By status
            status = record.new_status or 'unknown'
            wip_summary['by_status'][status] = wip_summary['by_status'].get(status, 0) + 1
            
            # By operation
            if record.operation_code:
                wip_summary['by_operation'][record.operation_code] = \
                    wip_summary['by_operation'].get(record.operation_code, 0) + 1
            
            # By work center
            if record.work_center:
                wip_summary['by_work_center'][record.work_center] = \
                    wip_summary['by_work_center'].get(record.work_center, 0) + 1
        
        return wip_summary
    
    @staticmethod
    def calculate_lot_genealogy(session, wip_lot_number):
        """Build complete genealogy including all operations and materials"""
        
        history = session.query(WIPHistory).filter_by(
            wip_lot_number=wip_lot_number
        ).order_by(WIPHistory.event_timestamp).all()
        
        if not history:
            return None
        
        first_event = history[0]
        last_event = history[-1]
        
        genealogy = {
            'lot_number': wip_lot_number,
            'work_order': first_event.work_order_number,
            'created': first_event.event_timestamp,
            'last_updated': last_event.event_timestamp,
            'current_status': last_event.new_status,
            'current_location': {
                'operation': last_event.operation_code,
                'equipment': last_event.equipment_code,
                'work_center': last_event.work_center
            },
            'operations_completed': [],
            'quality_events': [],
            'total_cycle_time': 0,
            'total_queue_time': 0
        }
        
        # Process history events
        for i, event in enumerate(history):
            if event.event_type == WIPEventType.PROCESS_COMPLETE.value:
                genealogy['operations_completed'].append({
                    'operation': event.operation_code,
                    'completed_at': event.event_timestamp,
                    'cycle_time': float(event.cycle_time_minutes) if event.cycle_time_minutes else 0,
                    'good_quantity': float(event.good_quantity) if event.good_quantity else 0,
                    'operator': event.operator_name or event.operator_id
                })
                if event.cycle_time_minutes:
                    genealogy['total_cycle_time'] += float(event.cycle_time_minutes)
            
            elif event.event_type in [WIPEventType.INSPECTION_PASS.value, 
                                     WIPEventType.INSPECTION_FAIL.value]:
                genealogy['quality_events'].append({
                    'timestamp': event.event_timestamp,
                    'result': event.quality_check_result,
                    'defects': event.defect_codes,
                    'measurements': event.measurement_data
                })
            
            elif event.event_type == WIPEventType.QUEUE_ENTRY.value and i > 0:
                # Calculate queue time from previous event
                queue_time = (event.event_timestamp - history[i-1].event_timestamp).total_seconds() / 60
                genealogy['total_queue_time'] += queue_time
        
        genealogy['total_manufacturing_time'] = genealogy['total_cycle_time'] + genealogy['total_queue_time']
        
        return genealogy


# ============================================================================
# SAMPLE DATA GENERATION
# ============================================================================

def generate_realistic_wip_history(session, num_work_orders=5, lots_per_order=3):
    """Generate realistic WIP history data for testing"""
    
    from manufacturing.datamodel import (
        WorkOrder, WIPRecord, Product, Route, Sequence, 
        Operation, Equipment, WorkOrderStatus
    )
    
    # Shifts for realistic data
    shifts = ['Day', 'Evening', 'Night']
    operators = [
        ('OP001', 'John Smith'),
        ('OP002', 'Jane Doe'),
        ('OP003', 'Bob Johnson'),
        ('OP004', 'Alice Brown')
    ]
    
    # Common defect codes
    defect_codes = ['SCRATCH', 'DENT', 'DIMENSION', 'FINISH', 'CONTAMINATION']
    
    # Get existing products and routes
    products = session.query(Product).limit(num_work_orders).all()
    
    for product in products:
        # Create work order
        wo_number = f"WO-{datetime.now().year}-{random.randint(1000, 9999)}"
        wo = WorkOrder(
            work_order_number=wo_number,
            product=product,
            route=product.routes[0] if product.routes else None,
            order_quantity=random.randint(100, 1000),
            due_date=datetime.now(local_tz) + timedelta(days=random.randint(7, 30)),
            priority=random.randint(1, 5),
            status=WorkOrderStatus.IN_PROGRESS,
            customer_name=f"Customer-{random.randint(1, 100)}"
        )
        session.add(wo)
        session.flush()
        
        # Create multiple lots for the work order
        lot_quantity = wo.order_quantity / lots_per_order
        
        for lot_num in range(lots_per_order):
            lot_number = f"LOT-{wo_number}-{lot_num + 1:03d}"
            
            # Create WIP record
            wip = WIPRecord(
                wip_lot_number=lot_number,
                work_order=wo,
                quantity=lot_quantity,
                status=WIPStatus.QUEUE,
                plant="Plant-1",
                work_center=f"WC-{random.randint(1, 5):02d}",
                created_at=datetime.now(local_tz) - timedelta(days=random.randint(1, 7))
            )
            session.add(wip)
            session.flush()
            
            # Generate history events
            current_time = wip.created_at
            
            # Get operations from route
            if wo.route and wo.route.sequences:
                operations = []
                for sequence in wo.route.sequences:
                    operations.extend(sequence.operations)
                
                for op_index, operation in enumerate(operations):
                    # Queue entry
                    current_time += timedelta(minutes=random.randint(10, 60))
                    wip.current_operation = operation
                    wip.status = WIPStatus.QUEUE
                    
                    queue_history = create_wip_history_entry(
                        wip,
                        WIPEventType.QUEUE_ENTRY.value,
                        timestamp=current_time,
                        operator_id=random.choice(operators)[0],
                        shift=random.choice(shifts)
                    )
                    session.add(queue_history)
                    
                    # Setup start (if first piece)
                    if lot_num == 0:
                        current_time += timedelta(minutes=random.randint(5, 30))
                        setup_history = create_wip_history_entry(
                            wip,
                            WIPEventType.SETUP_START.value,
                            timestamp=current_time,
                            operator_id=random.choice(operators)[0],
                            shift=random.choice(shifts)
                        )
                        session.add(setup_history)
                        
                        # Setup complete
                        setup_time = random.randint(15, 45)
                        current_time += timedelta(minutes=setup_time)
                        setup_complete = create_wip_history_entry(
                            wip,
                            WIPEventType.SETUP_COMPLETE.value,
                            timestamp=current_time,
                            setup_time_minutes=setup_time,
                            operator_id=random.choice(operators)[0],
                            shift=random.choice(shifts)
                        )
                        session.add(setup_complete)
                    
                    # Process start
                    current_time += timedelta(minutes=random.randint(5, 15))
                    wip.status = WIPStatus.IN_PROCESS
                    wip.process_start_time = current_time
                    
                    operator = random.choice(operators)
                    process_start = create_wip_history_entry(
                        wip,
                        WIPEventType.PROCESS_START.value,
                        timestamp=current_time,
                        operator_id=operator[0],
                        operator_name=operator[1],
                        shift=random.choice(shifts),
                        terminal_id=f"TERM-{random.randint(1, 10):02d}"
                    )
                    session.add(process_start)
                    
                    # Process complete
                    cycle_time = random.randint(10, 120)
                    current_time += timedelta(minutes=cycle_time)
                    
                    # Determine quality outcome
                    passed_quality = random.random() > 0.1  # 90% pass rate
                    good_qty = lot_quantity if passed_quality else lot_quantity * 0.95
                    scrap_qty = 0 if passed_quality else lot_quantity * 0.05
                    
                    wip.status = WIPStatus.COMPLETED
                    wip.process_end_time = current_time
                    wip.good_quantity = good_qty
                    wip.scrap_quantity = scrap_qty
                    
                    process_complete = create_wip_history_entry(
                        wip,
                        WIPEventType.PROCESS_COMPLETE.value,
                        timestamp=current_time,
                        cycle_time_minutes=cycle_time,
                        operator_id=operator[0],
                        operator_name=operator[1],
                        shift=random.choice(shifts)
                    )
                    session.add(process_complete)
                    
                    # Quality inspection (random, not every operation)
                    if random.random() > 0.5:
                        current_time += timedelta(minutes=random.randint(5, 15))
                        
                        measurements = {
                            'dimension_a': round(random.uniform(9.95, 10.05), 3),
                            'dimension_b': round(random.uniform(19.90, 20.10), 3),
                            'surface_finish': round(random.uniform(0.8, 1.2), 2)
                        }
                        
                        inspection_result = WIPEventType.INSPECTION_PASS if passed_quality else WIPEventType.INSPECTION_FAIL
                        defects = None if passed_quality else random.sample(defect_codes, random.randint(1, 2))
                        
                        inspection = create_wip_history_entry(
                            wip,
                            inspection_result.value,
                            timestamp=current_time,
                            quality_check_result='pass' if passed_quality else 'fail',
                            measurement_data=measurements,
                            defect_codes=defects,
                            operator_id=f"QC{random.randint(1, 3):02d}",
                            operator_name=f"Inspector-{random.randint(1, 3)}"
                        )
                        session.add(inspection)
                        
                        # If failed, add rework
                        if not passed_quality:
                            current_time += timedelta(minutes=random.randint(30, 120))
                            rework_start = create_wip_history_entry(
                                wip,
                                WIPEventType.REWORK_START.value,
                                timestamp=current_time,
                                reason_code=defects[0] if defects else 'GENERAL',
                                operator_id=operator[0]
                            )
                            session.add(rework_start)
                            
                            rework_time = random.randint(20, 60)
                            current_time += timedelta(minutes=rework_time)
                            rework_complete = create_wip_history_entry(
                                wip,
                                WIPEventType.REWORK_COMPLETE.value,
                                timestamp=current_time,
                                cycle_time_minutes=rework_time,
                                operator_id=operator[0]
                            )
                            session.add(rework_complete)
                    
                    # Move to next operation (if not last)
                    if op_index < len(operations) - 1:
                        current_time += timedelta(minutes=random.randint(5, 15))
                        move_start = create_wip_history_entry(
                            wip,
                            WIPEventType.MOVE_START.value,
                            timestamp=current_time
                        )
                        session.add(move_start)
                        
                        move_time = random.randint(5, 30)
                        current_time += timedelta(minutes=move_time)
                        move_complete = create_wip_history_entry(
                            wip,
                            WIPEventType.MOVE_COMPLETE.value,
                            timestamp=current_time,
                            move_time_minutes=move_time
                        )
                        session.add(move_complete)
                
                # Final completion
                current_time += timedelta(minutes=random.randint(10, 30))
                wip.status = WIPStatus.COMPLETED
                completion = create_wip_history_entry(
                    wip,
                    WIPEventType.COMPLETE.value,
                    timestamp=current_time,
                    notes=f"Lot completed successfully. Final quantity: {good_qty}"
                )
                session.add(completion)
    
    session.commit()
    print(f"Generated {num_work_orders * lots_per_order} WIP lots with detailed history")


# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

def example_queries():
    """Example SQL queries for WIP history analysis"""
    
    queries = """
    -- 1. Current status of all WIP lots
    WITH latest_status AS (
        SELECT 
            wip_lot_number,
            MAX(event_timestamp) as latest_timestamp
        FROM wip_history
        GROUP BY wip_lot_number
    )
    SELECT 
        h.wip_lot_number,
        h.work_order_number,
        h.new_status as current_status,
        h.operation_code as current_operation,
        h.equipment_code as current_equipment,
        h.event_timestamp as last_update,
        h.quantity,
        h.good_quantity
    FROM wip_history h
    JOIN latest_status ls ON h.wip_lot_number = ls.wip_lot_number 
        AND h.event_timestamp = ls.latest_timestamp
    ORDER BY h.event_timestamp DESC;
    
    -- 2. Calculate cycle time by operation
    SELECT 
        operation_code,
        COUNT(*) as lots_processed,
        AVG(cycle_time_minutes) as avg_cycle_time,
        MIN(cycle_time_minutes) as min_cycle_time,
        MAX(cycle_time_minutes) as max_cycle_time,
        STDDEV(cycle_time_minutes) as cycle_time_stddev
    FROM wip_history
    WHERE event_type = 'process_complete'
        AND cycle_time_minutes IS NOT NULL
        AND event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY operation_code
    ORDER BY operation_code;
    
    -- 3. First Pass Yield by operation
    SELECT 
        operation_code,
        COUNT(CASE WHEN quality_check_result = 'pass' THEN 1 END) as passed,
        COUNT(CASE WHEN quality_check_result = 'fail' THEN 1 END) as failed,
        ROUND(
            COUNT(CASE WHEN quality_check_result = 'pass' THEN 1 END) * 100.0 / 
            NULLIF(COUNT(*), 0), 
            2
        ) as first_pass_yield_pct
    FROM wip_history
    WHERE event_type IN ('inspection_pass', 'inspection_fail')
        AND event_timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY operation_code
    ORDER BY first_pass_yield_pct DESC;
    
    -- 4. Lot genealogy - complete path through factory
    SELECT 
        wip_lot_number,
        event_timestamp,
        event_type,
        operation_code,
        equipment_code,
        operator_name,
        quantity,
        good_quantity,
        cycle_time_minutes,
        quality_check_result,
        notes
    FROM wip_history
    WHERE wip_lot_number = 'LOT-001'
    ORDER BY event_timestamp;
    
    -- 5. Real-time WIP aging report
    WITH current_locations AS (
        SELECT DISTINCT ON (wip_lot_number)
            wip_lot_number,
            work_order_number,
            operation_code,
            event_timestamp as last_movement,
            new_status
        FROM wip_history
        ORDER BY wip_lot_number, event_timestamp DESC
    )
    SELECT 
        wip_lot_number,
        work_order_number,
        operation_code,
        new_status,
        last_movement,
        EXTRACT(EPOCH FROM (NOW() - last_movement))/3600 as hours_since_movement,
        CASE 
            WHEN EXTRACT(EPOCH FROM (NOW() - last_movement))/3600 > 24 THEN 'CRITICAL'
            WHEN EXTRACT(EPOCH FROM (NOW() - last_movement))/3600 > 8 THEN 'WARNING'
            ELSE 'OK'
        END as aging_status
    FROM current_locations
    WHERE new_status NOT IN ('complete', 'cancelled', 'scrapped')
    ORDER BY hours_since_movement DESC;
    
    -- 6. Operator performance metrics
    SELECT 
        operator_name,
        COUNT(DISTINCT wip_lot_number) as lots_processed,
        AVG(cycle_time_minutes) as avg_cycle_time,
        SUM(good_quantity) as total_good_qty,
        SUM(scrap_quantity) as total_scrap_qty,
        ROUND(
            SUM(good_quantity) * 100.0 / 
            NULLIF(SUM(good_quantity) + SUM(scrap_quantity), 0),
            2
        ) as yield_pct
    FROM wip_history
    WHERE event_type = 'process_complete'
        AND operator_name IS NOT NULL
        AND event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY operator_name
    ORDER BY lots_processed DESC;
    
    -- 7. Bottleneck analysis - operations with longest queue times
    WITH queue_times AS (
        SELECT 
            h1.wip_lot_number,
            h1.operation_code,
            h1.event_timestamp as queue_entry,
            MIN(h2.event_timestamp) as process_start,
            EXTRACT(EPOCH FROM (MIN(h2.event_timestamp) - h1.event_timestamp))/60 as queue_minutes
        FROM wip_history h1
        JOIN wip_history h2 
            ON h1.wip_lot_number = h2.wip_lot_number
            AND h2.event_type = 'process_start'
            AND h2.event_timestamp > h1.event_timestamp
        WHERE h1.event_type = 'queue_entry'
        GROUP BY h1.wip_lot_number, h1.operation_code, h1.event_timestamp
    )
    SELECT 
        operation_code,
        COUNT(*) as lots_queued,
        AVG(queue_minutes) as avg_queue_time,
        MAX(queue_minutes) as max_queue_time,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY queue_minutes) as median_queue_time
    FROM queue_times
    WHERE queue_minutes IS NOT NULL
    GROUP BY operation_code
    ORDER BY avg_queue_time DESC;
    """
    
    return queries


if __name__ == "__main__":
    # Create tables if they don't exist
    from manufacturing.datamodel import create_database, get_session
    
    engine = create_database('postgresql://myuser:mypassword@localhost:5433/manufacturing')
    session = get_session(engine)
    
    # Generate sample history data
    generate_realistic_wip_history(session, num_work_orders=5, lots_per_order=3)
    
    # Example: Track a lot through operations
    tracker = WIPTracker()
    
    # Start an operation
    tracker.start_operation(session, "LOT-001", operator_id="OP001", terminal_id="TERM-01")
    
    # Complete the operation
    tracker.complete_operation(session, "LOT-001", good_qty=45, scrap_qty=5, 
                              defect_codes=['SCRATCH'], operator_id="OP001")
    
    # Move to next operation
    tracker.move_to_next_operation(session, "LOT-001", next_operation_id=2, 
                                  move_time_minutes=15)
    
    # Record quality check
    tracker.record_quality_check(session, "LOT-001", passed=True, 
                                measurements={'dimension': 10.05}, 
                                inspector_id="QC01")
    
    # Get analytics
    analytics = WIPAnalytics()
    
    # Get lot timeline
    timeline = analytics.get_lot_timeline(session, "LOT-001")
    print("Lot Timeline:", json.dumps(timeline, indent=2, default=str))
    
    # Get operation metrics
    metrics = analytics.calculate_operation_metrics(
        session, 
        operation_id=1,
        start_date=datetime.now(local_tz) - timedelta(days=7),
        end_date=datetime.now(local_tz)
    )
    print("Operation Metrics:", json.dumps(metrics, indent=2))
    
    # Get WIP status
    wip_status = analytics.get_current_wip_status(session)
    print("Current WIP Status:", json.dumps(wip_status, indent=2))
    
    print("\nWIP History tracking system created successfully!")
    print("\nExample SQL queries saved to: wip_history_queries.sql")
    
    # Save queries to file
    with open('wip_history_queries.sql', 'w') as f:
        f.write(example_queries())