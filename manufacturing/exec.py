from manufacturing.datamodel import create_database, get_session, Product, Route, Operation, Machine, WorkCenter, Material, BillOfMaterials, UniitOfMeasure, Equipment, EquipmentStatus, Sequence, WorkOrder, WorkOrderStatus, WIPRecord, WIPStatus, MaterialInventory, MaterialType, BOMItem, operation_material
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# ============================================================================
# Example usage
# ============================================================================  

#connection_string='sqlite:///manufacturing.db'

connection_string='postgresql://myuser:mypassword@localhost:5433/manufacturing'

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