"""
Manufacturing System Data Model using SQLAlchemy
Modules: PRSOE, BOM, Work Orders, WIP
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Table, Text, DECIMAL, Date, JSON, UniqueConstraint,
    CheckConstraint, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.hybrid import hybrid_property

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_postgres_database(dbname, user, password, host='localhost', port=5432):
    # Connect to the default database
    conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    # Check if database exists
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{dbname}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f'CREATE DATABASE "{dbname}"')
        print(f"Database '{dbname}' created.")
    else:
        print(f"Database '{dbname}' already exists.")
    cur.close()
    conn.close()

# Usage example:
create_postgres_database(
    dbname='manufacturing',
    user='myuser',
    password='mypassword',
    host='localhost',
    port=5433
)


Base = declarative_base()

# ============================================================================
# ENUMS
# ============================================================================

class EquipmentStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    BROKEN = "broken"
    RETIRED = "retired"

class MaterialType(Enum):
    DIRECT = "direct"      # Direct materials (part of final product)
    INDIRECT = "indirect"  # Indirect materials (consumables, tools)
    
class OperationStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"

class WorkOrderStatus(Enum):
    DRAFT = "draft"
    PLANNED = "planned"
    RELEASED = "released"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"

class WIPStatus(Enum):
    QUEUE = "queue"
    IN_PROCESS = "in_process"
    COMPLETED = "completed"
    SCRAPPED = "scrapped"
    REWORK = "rework"

class UnitOfMeasure(Enum):
    PIECE = "piece"
    KG = "kg"
    LITER = "liter"
    METER = "meter"
    HOUR = "hour"
    UNIT = "unit"

# ============================================================================
# ASSOCIATION TABLES
# ============================================================================

# Many-to-many relationship between Operation and Equipment
operation_equipment = Table(
    'operation_equipment',
    Base.metadata,
    Column('operation_id', Integer, ForeignKey('operations.id', ondelete='CASCADE')),
    Column('equipment_id', Integer, ForeignKey('equipment.id', ondelete='CASCADE')),
    Column('is_primary', Boolean, default=False),
    Column('setup_time_minutes', Float, default=0),
    Column('capacity_per_hour', Float),
    UniqueConstraint('operation_id', 'equipment_id')
)

# Many-to-many relationship between Operation and Material
operation_material = Table(
    'operation_material',
    Base.metadata,
    Column('operation_id', Integer, ForeignKey('operations.id', ondelete='CASCADE')),
    Column('material_id', Integer, ForeignKey('materials.id', ondelete='CASCADE')),
    Column('quantity_per_unit', DECIMAL(10, 4), nullable=False),
    Column('material_type', SQLEnum(MaterialType), nullable=False),
    Column('scrap_rate', DECIMAL(5, 2), default=0),  # Percentage
    UniqueConstraint('operation_id', 'material_id')
)

# Substitution relationships for operations within a sequence
operation_substitution = Table(
    'operation_substitution',
    Base.metadata,
    Column('primary_operation_id', Integer, ForeignKey('operations.id', ondelete='CASCADE')),
    Column('substitute_operation_id', Integer, ForeignKey('operations.id', ondelete='CASCADE')),
    Column('sequence_id', Integer, ForeignKey('sequences.id', ondelete='CASCADE')),
    Column('priority', Integer, default=1),  # Lower number = higher priority
    Column('efficiency_factor', DECIMAL(5, 2), default=100),  # Percentage
    UniqueConstraint('primary_operation_id', 'substitute_operation_id', 'sequence_id')
)

# ============================================================================
# PRSOE MODULE
# ============================================================================

class Equipment(Base):
    """E - Equipment in PRSOE"""
    __tablename__ = 'equipment'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    equipment_group = Column(String(50))
    
    # Capacity and performance
    capacity_per_hour = Column(DECIMAL(10, 2))
    efficiency = Column(DECIMAL(5, 2), default=85.0)  # Percentage
    oee_target = Column(DECIMAL(5, 2), default=85.0)  # Overall Equipment Effectiveness
    
    # Status and maintenance
    status = Column(SQLEnum(EquipmentStatus), default=EquipmentStatus.AVAILABLE)
    last_maintenance_date = Column(DateTime)
    next_maintenance_date = Column(DateTime)
    mtbf_hours = Column(Float)  # Mean Time Between Failures
    mttr_hours = Column(Float)  # Mean Time To Repair
    
    # Location
    plant = Column(String(50))
    department = Column(String(50))
    work_center = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    operations = relationship('Operation', secondary=operation_equipment, back_populates='equipment')
    wip_records = relationship('WIPRecord', back_populates='equipment')
    
    def __repr__(self):
        return f"<Equipment {self.code}: {self.name}>"


class Product(Base):
    """P - Product in PRSOE"""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    part_number = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    revision = Column(String(10), default='A')
    
    # Product attributes
    product_family = Column(String(50))
    product_type = Column(String(50))
    unit_of_measure = Column(SQLEnum(UnitOfMeasure), default=UnitOfMeasure.PIECE)
    weight = Column(DECIMAL(10, 3))
    volume = Column(DECIMAL(10, 3))
    
    # Cost information
    standard_cost = Column(DECIMAL(10, 2))
    target_cost = Column(DECIMAL(10, 2))
    
    # Quality specifications
    specifications = Column(JSON)  # Store quality specs as JSON
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    routes = relationship('Route', back_populates='product')
    bom_items = relationship('BOMItem', foreign_keys='BOMItem.parent_product_id', back_populates='parent_product')
    work_orders = relationship('WorkOrder', back_populates='product')
    
    def __repr__(self):
        return f"<Product {self.part_number}: {self.name}>"


class Route(Base):
    """R - Route in PRSOE"""
    __tablename__ = 'routes'
    
    id = Column(Integer, primary_key=True)
    route_code = Column(String(50), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Route attributes
    version = Column(Integer, default=1)
    is_primary = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    effective_date = Column(Date)
    expiration_date = Column(Date)
    
    # Performance metrics
    total_cycle_time = Column(DECIMAL(10, 2))  # Total time in minutes
    total_labor_time = Column(DECIMAL(10, 2))
    total_machine_time = Column(DECIMAL(10, 2))
    
    # Foreign keys
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship('Product', back_populates='routes')
    sequences = relationship('Sequence', back_populates='route', cascade='all, delete-orphan')
    work_orders = relationship('WorkOrder', back_populates='route')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('product_id', 'route_code', 'version'),
        Index('idx_route_product', 'product_id'),
    )
    
    def __repr__(self):
        return f"<Route {self.route_code} v{self.version}>"


class Sequence(Base):
    """S - Sequence in PRSOE"""
    __tablename__ = 'sequences'
    
    id = Column(Integer, primary_key=True)
    sequence_number = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Sequence attributes
    is_parallel = Column(Boolean, default=False)  # Can run in parallel with other sequences
    is_optional = Column(Boolean, default=False)
    
    # Time estimates
    setup_time = Column(DECIMAL(10, 2), default=0)
    cycle_time = Column(DECIMAL(10, 2))
    queue_time = Column(DECIMAL(10, 2), default=0)
    move_time = Column(DECIMAL(10, 2), default=0)
    
    # Foreign keys
    route_id = Column(Integer, ForeignKey('routes.id', ondelete='CASCADE'))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    route = relationship('Route', back_populates='sequences')
    operations = relationship('Operation', back_populates='sequence', cascade='all, delete-orphan')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('route_id', 'sequence_number'),
        Index('idx_sequence_route', 'route_id'),
    )
    
    @hybrid_property
    def total_time(self):
        """Calculate total time for the sequence"""
        return (self.setup_time or 0) + (self.cycle_time or 0) + \
               (self.queue_time or 0) + (self.move_time or 0)
    
    def __repr__(self):
        return f"<Sequence {self.sequence_number}: {self.name}>"


class Operation(Base):
    """O - Operation in PRSOE"""
    __tablename__ = 'operations'
    
    id = Column(Integer, primary_key=True)
    operation_number = Column(Integer, nullable=False)
    operation_code = Column(String(50), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Operation type and status
    operation_type = Column(String(50))  # e.g., 'machining', 'assembly', 'inspection'
    status = Column(SQLEnum(OperationStatus), default=OperationStatus.ACTIVE)
    
    # Time standards
    setup_time_minutes = Column(DECIMAL(10, 2), default=0)
    cycle_time_minutes = Column(DECIMAL(10, 2))
    labor_time_minutes = Column(DECIMAL(10, 2))
    machine_time_minutes = Column(DECIMAL(10, 2))
    
    # Resource requirements
    operators_required = Column(Integer, default=1)
    skill_level_required = Column(String(20))
    
    # Quality parameters
    inspection_required = Column(Boolean, default=False)
    inspection_frequency = Column(Integer)  # Every N pieces
    scrap_rate = Column(DECIMAL(5, 2), default=0)  # Percentage
    rework_rate = Column(DECIMAL(5, 2), default=0)  # Percentage
    
    # Work instructions
    work_instructions = Column(Text)
    setup_instructions = Column(Text)
    quality_instructions = Column(Text)
    
    # Foreign keys
    sequence_id = Column(Integer, ForeignKey('sequences.id', ondelete='CASCADE'))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sequence = relationship('Sequence', back_populates='operations')
    equipment = relationship('Equipment', secondary=operation_equipment, back_populates='operations')
    materials = relationship('Material', secondary=operation_material, back_populates='operations')
    bom_items = relationship('BOMItem', back_populates='operation')
    
    # Substitution relationships
    primary_for = relationship(
        'Operation',
        secondary=operation_substitution,
        primaryjoin=id == operation_substitution.c.substitute_operation_id,
        secondaryjoin=id == operation_substitution.c.primary_operation_id,
        backref='substitutes'
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('sequence_id', 'operation_number'),
        Index('idx_operation_sequence', 'sequence_id'),
    )
    
    def __repr__(self):
        return f"<Operation {self.operation_code}: {self.name}>"


# ============================================================================
# MATERIALS MODULE
# ============================================================================

class Material(Base):
    """Materials used in operations"""
    __tablename__ = 'materials'
    
    id = Column(Integer, primary_key=True)
    material_code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Material classification
    material_type = Column(SQLEnum(MaterialType), nullable=False)
    material_group = Column(String(50))
    
    # Units and conversion
    unit_of_measure = Column(SQLEnum(UnitOfMeasure), default=UnitOfMeasure.PIECE)
    unit_cost = Column(DECIMAL(10, 4))
    
    # Inventory parameters
    min_stock_level = Column(DECIMAL(10, 2), default=0)
    max_stock_level = Column(DECIMAL(10, 2))
    reorder_point = Column(DECIMAL(10, 2))
    lead_time_days = Column(Integer)
    
    # Quality
    shelf_life_days = Column(Integer)
    batch_tracked = Column(Boolean, default=False)
    serial_tracked = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    operations = relationship('Operation', secondary=operation_material, back_populates='materials')
    inventory_records = relationship('MaterialInventory', back_populates='material')
    
    def __repr__(self):
        return f"<Material {self.material_code}: {self.name}>"


# ============================================================================
# BOM MODULE
# ============================================================================

class BOMItem(Base):
    """Bill of Materials items"""
    __tablename__ = 'bom_items'
    
    id = Column(Integer, primary_key=True)
    
    # BOM hierarchy
    parent_product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'))
    component_product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'))
    
    # BOM details
    quantity_per = Column(DECIMAL(10, 4), nullable=False)
    unit_of_measure = Column(SQLEnum(UnitOfMeasure))
    
    # Operation link
    operation_id = Column(Integer, ForeignKey('operations.id'))
    
    # Effectivity
    effective_date = Column(Date)
    expiration_date = Column(Date)
    
    # BOM type
    is_phantom = Column(Boolean, default=False)  # Phantom BOM
    is_alternate = Column(Boolean, default=False)
    
    # Scrap and yield
    scrap_factor = Column(DECIMAL(5, 2), default=0)
    yield_percentage = Column(DECIMAL(5, 2), default=100)
    
    # Reference info
    reference_designator = Column(String(100))
    find_number = Column(String(20))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parent_product = relationship('Product', foreign_keys=[parent_product_id], back_populates='bom_items')
    component_product = relationship('Product', foreign_keys=[component_product_id])
    operation = relationship('Operation', back_populates='bom_items')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('parent_product_id', 'component_product_id', 'operation_id'),
        Index('idx_bom_parent', 'parent_product_id'),
        Index('idx_bom_component', 'component_product_id'),
        CheckConstraint('parent_product_id != component_product_id', name='check_no_self_reference'),
    )
    
    def __repr__(self):
        return f"<BOMItem Parent:{self.parent_product_id} Component:{self.component_product_id}>"


# ============================================================================
# WORK ORDERS MODULE
# ============================================================================

class WorkOrder(Base):
    """Manufacturing work orders"""
    __tablename__ = 'work_orders'
    
    id = Column(Integer, primary_key=True)
    work_order_number = Column(String(50), unique=True, nullable=False, index=True)
    
    # Order details
    product_id = Column(Integer, ForeignKey('products.id'))
    route_id = Column(Integer, ForeignKey('routes.id'))
    
    # Quantities
    order_quantity = Column(DECIMAL(10, 2), nullable=False)
    completed_quantity = Column(DECIMAL(10, 2), default=0)
    scrapped_quantity = Column(DECIMAL(10, 2), default=0)
    
    # Scheduling
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest
    planned_start_date = Column(DateTime)
    planned_end_date = Column(DateTime)
    actual_start_date = Column(DateTime)
    actual_end_date = Column(DateTime)
    due_date = Column(DateTime, nullable=False)
    
    # Status
    status = Column(SQLEnum(WorkOrderStatus), default=WorkOrderStatus.DRAFT)
    
    # Parent/Child work orders
    parent_work_order_id = Column(Integer, ForeignKey('work_orders.id'))
    
    # Customer/Sales reference
    sales_order_number = Column(String(50))
    customer_name = Column(String(100))
    
    # Cost tracking
    estimated_cost = Column(DECIMAL(10, 2))
    actual_cost = Column(DECIMAL(10, 2))
    
    # Notes
    notes = Column(Text)
    
    # Metadata
    created_by = Column(String(50))
    released_by = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship('Product', back_populates='work_orders')
    route = relationship('Route', back_populates='work_orders')
    parent_work_order = relationship('WorkOrder', remote_side=[id], backref='child_work_orders')
    wip_records = relationship('WIPRecord', back_populates='work_order')
    
    @hybrid_property
    def completion_percentage(self):
        """Calculate completion percentage"""
        if self.order_quantity:
            return (self.completed_quantity / self.order_quantity) * 100
        return 0
    
    def __repr__(self):
        return f"<WorkOrder {self.work_order_number}>"


# ============================================================================
# WIP MODULE
# ============================================================================

class WIPRecord(Base):
    """Work In Progress tracking"""
    __tablename__ = 'wip_records'
    
    id = Column(Integer, primary_key=True)
    wip_lot_number = Column(String(50), unique=True, nullable=False, index=True)
    
    # Work order reference
    work_order_id = Column(Integer, ForeignKey('work_orders.id'))
    
    # Current location in process
    current_operation_id = Column(Integer, ForeignKey('operations.id'))
    current_equipment_id = Column(Integer, ForeignKey('equipment.id'))
    
    # Quantities
    quantity = Column(DECIMAL(10, 2), nullable=False)
    good_quantity = Column(DECIMAL(10, 2))
    scrap_quantity = Column(DECIMAL(10, 2), default=0)
    rework_quantity = Column(DECIMAL(10, 2), default=0)
    
    # Status
    status = Column(SQLEnum(WIPStatus), default=WIPStatus.QUEUE)
    
    # Timestamps
    queue_entry_time = Column(DateTime)
    process_start_time = Column(DateTime)
    process_end_time = Column(DateTime)
    
    # Location
    plant = Column(String(50))
    work_center = Column(String(50))
    storage_location = Column(String(50))
    
    # Batch/Serial tracking
    batch_number = Column(String(50))
    serial_numbers = Column(JSON)  # Array of serial numbers if applicable
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    work_order = relationship('WorkOrder', back_populates='wip_records')
    current_operation = relationship('Operation')
    equipment = relationship('Equipment', back_populates='wip_records')
    
    @hybrid_property
    def cycle_time(self):
        """Calculate actual cycle time"""
        if self.process_start_time and self.process_end_time:
            delta = self.process_end_time - self.process_start_time
            return delta.total_seconds() / 60  # Return in minutes
        return None
    
    def __repr__(self):
        return f"<WIPRecord {self.wip_lot_number}>"


class MaterialInventory(Base):
    """Material inventory tracking"""
    __tablename__ = 'material_inventory'
    
    id = Column(Integer, primary_key=True)
    
    # Material reference
    material_id = Column(Integer, ForeignKey('materials.id'))
    
    # Location
    warehouse = Column(String(50), nullable=False)
    location = Column(String(50))
    
    # Quantities
    quantity_on_hand = Column(DECIMAL(10, 2), nullable=False, default=0)
    quantity_available = Column(DECIMAL(10, 2), nullable=False, default=0)
    quantity_allocated = Column(DECIMAL(10, 2), default=0)
    quantity_on_order = Column(DECIMAL(10, 2), default=0)
    
    # Batch/Lot tracking
    batch_number = Column(String(50))
    lot_number = Column(String(50))
    expiration_date = Column(Date)
    
    # Cost
    unit_cost = Column(DECIMAL(10, 4))
    total_value = Column(DECIMAL(12, 2))
    
    # Last transactions
    last_receipt_date = Column(DateTime)
    last_issue_date = Column(DateTime)
    last_count_date = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    material = relationship('Material', back_populates='inventory_records')
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('material_id', 'warehouse', 'location', 'batch_number'),
        Index('idx_inventory_material', 'material_id'),
        Index('idx_inventory_warehouse', 'warehouse'),
    )
    
    def __repr__(self):
        return f"<MaterialInventory Material:{self.material_id} Warehouse:{self.warehouse}>"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_database(connection_string='sqlite:///manufacturing.db'):
    """Create database with all tables"""
    engine = create_engine(connection_string, echo=True)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()


from sqlalchemy import text

def fetch_pg_tables(engine):
    # Test the connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database()"))
        print(f"Connected to database: {result.scalar()}")
        
        result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
        tables = result.fetchall()
        print(f"Tables created: {[t[0] for t in tables]}")
        
    return tables

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # connection_string='sqlite:///manufacturing.db'

    # Becasue the postgresql is run via kind's cluster, use kubectl to get the CLUSTER-IP and PORT (usually 5432) to update the connection string
    # % kubectl get svc postgresql
    #  NAME         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
    #  postgresql   ClusterIP   10.96.5.165   <none>        5432/TCP   177m
    # 
    connection_string='postgresql://myuser:mypassword@localhost:5433/manufacturing'
    # Create database
    engine = create_database(connection_string=connection_string)
    session = get_session(engine)
    
    # Example: Create a product with route and operations
    
    # Create Product
    product = Product(
        part_number="PROD-001",
        name="Example Product",
        description="An example manufactured product",
        product_family="Family A",
        unit_of_measure=UnitOfMeasure.PIECE
    )
    session.add(product)
    
    # Create Equipment
    equipment1 = Equipment(
        code="CNC-001",
        name="CNC Machine 1",
        equipment_group="CNC",
        capacity_per_hour=10,
        status=EquipmentStatus.AVAILABLE,
        plant="Plant 1",
        department="Machining"
    )
    session.add(equipment1)
    
    # Create Route
    route = Route(
        route_code="RT-001",
        name="Primary Route",
        product=product,
        version=1,
        is_primary=True,
        is_active=True
    )
    session.add(route)
    
    # Create Sequence
    sequence = Sequence(
        sequence_number=10,
        name="Machining Sequence",
        route=route,
        setup_time=30,
        cycle_time=120
    )
    session.add(sequence)
    
    # Create Operation
    operation = Operation(
        operation_number=10,
        operation_code="OP-001",
        name="CNC Machining",
        sequence=sequence,
        operation_type="machining",
        setup_time_minutes=30,
        cycle_time_minutes=10,
        operators_required=1
    )
    operation.equipment.append(equipment1)
    session.add(operation)
    
    # Create Material
    material = Material(
        material_code="MAT-001",
        name="Raw Material A",
        material_type=MaterialType.DIRECT,
        unit_of_measure=UnitOfMeasure.KG,
        unit_cost=10.50
    )
    session.add(material)
    
    # Link material to operation (through association table)
    session.execute(
        operation_material.insert().values(
            operation_id=operation.id,
            material_id=material.id,
            quantity_per_unit=2.5,
            material_type=MaterialType.DIRECT
        )
    )
    
    # Create BOM item
    component_product = Product(
        part_number="COMP-001",
        name="Component A",
        unit_of_measure=UnitOfMeasure.PIECE
    )
    session.add(component_product)
    
    bom_item = BOMItem(
        parent_product=product,
        component_product=component_product,
        quantity_per=2,
        operation=operation,
        unit_of_measure=UnitOfMeasure.PIECE
    )
    session.add(bom_item)
    
    # Create Work Order
    work_order = WorkOrder(
        work_order_number="WO-2024-001",
        product=product,
        route=route,
        order_quantity=100,
        due_date=datetime(2024, 12, 31),
        priority=3,
        status=WorkOrderStatus.PLANNED
    )
    session.add(work_order)
    
    # Create WIP Record
    wip_record = WIPRecord(
        wip_lot_number="LOT-001",
        work_order=work_order,
        current_operation=operation,
        equipment=equipment1,
        quantity=50,
        status=WIPStatus.IN_PROCESS,
        plant="Plant 1",
        work_center="WC-001"
    )
    session.add(wip_record)
    
    # Create Material Inventory
    inventory = MaterialInventory(
        material=material,
        warehouse="WH-001",
        location="A-1-1",
        quantity_on_hand=1000,
        quantity_available=800,
        quantity_allocated=200,
        unit_cost=10.50
    )
    session.add(inventory)
    
    # Commit all changes
    session.commit()
    
    print("Manufacturing system data model created successfully!")
    
    # Example query: Find all operations for a product
    product_operations = session.query(Operation).join(
        Sequence
    ).join(
        Route
    ).filter(
        Route.product_id == product.id
    ).all()
    
    print(f"\nOperations for product {product.part_number}:")
    for op in product_operations:
        print(f"  - {op.operation_code}: {op.name}")
    
    session.close()