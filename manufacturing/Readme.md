# Data Model 

## Manufacturing Data Model
[Manufacturing Data Model](./manufacturing.py)
A comprehensive SQLAlchemy data model for manufacturing system. Here are the key features:
Model Structure
1. PRSOE Module

Product: Part numbers, specifications, cost information
Route: Version-controlled routing with multiple sequences per product
Sequence: Groups of operations that can be parallel or sequential
Operation: Individual work steps with time standards and resource requirements
Equipment: Machines/resources with capacity, status, and maintenance tracking
Material: Direct and indirect materials with inventory parameters

2. Key Relationships

Hierarchical: Route → Sequences → Operations
Substitution: Operations can have substitutes within a sequence (via operation_substitution table)
Resources: Operations link to both equipment and materials through association tables
Many-to-Many: Operations can use multiple equipment/materials, and equipment/materials can be used by multiple operations

3. BOM Module

BOMItem: Links parent products to component products
Operation Key: Each BOM item can reference the specific operation where it's consumed
Features: Effectivity dates, phantom BOMs, scrap factors, yield percentages

4. Work Orders Module

WorkOrder: Manufacturing instructions with:

Assigned quantities
Product and route references
Due dates and priority
Status tracking (Draft → Planned → Released → In Progress → Completed)
Parent-child relationships for hierarchical work orders
Cost tracking (estimated vs actual)



5. WIP Module

WIPRecord: Tracks work in progress through the factory

Current operation and equipment
Quantity tracking (good, scrap, rework)
Time stamps for cycle time calculation
Batch/serial number tracking


MaterialInventory: Manages material inventory

Multi-warehouse/location support
Quantity states (on-hand, available, allocated, on-order)
Batch/lot tracking with expiration dates



Special Features

Substitution Logic: The model supports operation substitutions within sequences with priority and efficiency factors
Time Tracking: Comprehensive time modeling including setup, cycle, queue, and move times
Quality Parameters: Scrap rates, rework rates, inspection requirements
Status Management: Enums for equipment status, operation status, work order status, and WIP status
Performance Metrics: Built-in properties for calculating completion percentages, cycle times, and total times
Constraints & Indexes: Proper unique constraints, foreign keys, and indexes for performance

Usage Example
The code includes a complete example showing how to:

Create products with routes, sequences, and operations
Link equipment and materials to operations
Create BOMs with operation keys
Issue work orders
Track WIP and material inventory

Database Creation
Simply run:
pythonengine = create_database('postgresql://user:pass@localhost/manufacturing')
# or for SQLite:
engine = create_database('sqlite:///manufacturing.db')
This model provides a solid foundation for a manufacturing execution system (MES) with full traceability and resource management capabilities.