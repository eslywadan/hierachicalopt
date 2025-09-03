from manufacturing.datamodel import WIPRecord, WorkOrder, Material, \
        operation_material, get_session, WIPStatus, Operation, Sequence
from datetime import datetime
from sqlalchemy import create_engine, select
import json


def get_lots_for_work_order(session, work_order_number,_print=False):
    """Get all WIP lots associated with a work order"""
    
    lots = session.query(WIPRecord).join(
        WorkOrder
    ).filter(
        WorkOrder.work_order_number == work_order_number
    ).all()
    if _print:
        for lot in lots:
            print(f"Lot: {lot.wip_lot_number}")
            print(f"  Status: {lot.status}")
            print(f"  Quantity: {lot.quantity}")
            print(f"  Current Operation: {lot.current_operation.name if lot.current_operation else 'N/A'}")
            print(f"  Equipment: {lot.equipment.name if lot.equipment else 'N/A'}")
        
    return lots

# Usage
connection_string='postgresql://myuser:mypassword@localhost:5433/manufacturing'
engine=create_engine(connection_string, echo=True)
session=get_session(engine)

def trace_lot_to_work_order(session, lot_number):
    """Trace a WIP lot back to its work order and product"""
    
    wip = session.query(WIPRecord).filter(
        WIPRecord.wip_lot_number == lot_number
    ).first()
    
    if wip:
        wo = wip.work_order
        product = wo.product if wo else None
        
        print(f"Lot: {wip.wip_lot_number}")
        print(f"├── Work Order: {wo.work_order_number if wo else 'N/A'}")
        print(f"├── Product: {product.part_number if product else 'N/A'} - {product.name if product else 'N/A'}")
        print(f"├── Order Quantity: {wo.order_quantity if wo else 'N/A'}")
        print(f"├── Customer: {wo.customer_name if wo else 'N/A'}")
        print(f"└── Sales Order: {wo.sales_order_number if wo else 'N/A'}")
        
        return wip, wo, product
    
    return None, None, None

# Usage
trace_lot_to_work_order(session, "LOT-001")

def get_lot_genealogy(session, lot_number):
    """Get complete genealogy for a lot including materials and operations"""
    
    wip = session.query(WIPRecord).filter(
        WIPRecord.wip_lot_number == lot_number
    ).first()
    
    if not wip:
        return None
    
    genealogy = {
        'lot_number': wip.wip_lot_number,
        'work_order': wip.work_order.work_order_number,
        'product': {
            'part_number': wip.work_order.product.part_number,
            'name': wip.work_order.product.name,
            'revision': wip.work_order.product.revision
        },
        'current_status': {
            'status': wip.status.value,
            'operation': wip.current_operation.operation_code if wip.current_operation else None,
            'equipment': wip.equipment.code if wip.equipment else None,
            'location': wip.work_center
        },
        'quantities': {
            'original': float(wip.quantity),
            'good': float(wip.good_quantity or 0),
            'scrap': float(wip.scrap_quantity or 0),
            'rework': float(wip.rework_quantity or 0)
        },
        'timeline': {
            'created': wip.created_at,
            'queue_entry': wip.queue_entry_time,
            'process_start': wip.process_start_time,
            'process_end': wip.process_end_time
        }
    }
    
    # Get materials consumed (if tracked)
    if wip.current_operation:
        materials = session.execute(
            select(Material, operation_material.c.quantity_per_unit)
            .join(operation_material)
            .where(operation_material.c.operation_id == wip.current_operation.id)
        ).all()
        
        genealogy['materials_consumed'] = [
            {
                'material_code': mat.material_code,
                'quantity': float(qty * wip.quantity)
            }
            for mat, qty in materials
        ]
    
    return genealogy

# Usage
genealogy = get_lot_genealogy(session, "LOT-001")
print(json.dumps(genealogy, indent=2, default=str))


class LotTracker:
    """Track lot movement through manufacturing process"""
    
    @staticmethod
    def get_lot_history(session, lot_number):
        """Get the complete movement history of a lot"""
        
        # In a real system, you'd have a WIPHistory table
        # For now, we'll show the current status
        wip = session.query(WIPRecord).filter(
            WIPRecord.wip_lot_number == lot_number
        ).first()
        
        if not wip:
            return None
            
        # Get the route and all operations
        route = wip.work_order.route
        operations = session.query(Operation).join(
            Sequence
        ).filter(
            Sequence.route_id == route.id
        ).order_by(
            Sequence.sequence_number,
            Operation.operation_number
        ).all()
        
        history = {
            'lot_number': lot_number,
            'current_operation': wip.current_operation.operation_code if wip.current_operation else None,
            'route': route.route_code,
            'operations_sequence': [
                {
                    'sequence': op.sequence.sequence_number,
                    'operation': op.operation_code,
                    'name': op.name,
                    'status': 'completed' if op.operation_number < (wip.current_operation.operation_number if wip.current_operation else 0) else
                             'in_process' if op == wip.current_operation else 
                             'pending'
                }
                for op in operations
            ]
        }
        
        return history
    
    @staticmethod
    def move_lot_to_next_operation(session, lot_number):
        """Move a lot to the next operation in sequence"""
        
        wip = session.query(WIPRecord).filter(
            WIPRecord.wip_lot_number == lot_number
        ).first()
        
        if not wip or not wip.current_operation:
            return False
        
        # Find next operation
        current_op = wip.current_operation
        next_op = session.query(Operation).join(
            Sequence
        ).filter(
            Sequence.route_id == wip.work_order.route.id,
            Sequence.sequence_number >= current_op.sequence.sequence_number,
            Operation.operation_number > current_op.operation_number
        ).order_by(
            Sequence.sequence_number,
            Operation.operation_number
        ).first()
        
        if next_op:
            # Update WIP record
            wip.current_operation = next_op
            wip.status = WIPStatus.QUEUE
            wip.queue_entry_time = datetime.utcnow()
            wip.process_start_time = None
            wip.process_end_time = None
            
            session.commit()
            return True
        
        return False
    
def get_work_order_lot_summary(session, work_order_number):
    """Get summary of all lots for a work order"""
    
    wo = session.query(WorkOrder).filter(
        WorkOrder.work_order_number == work_order_number
    ).first()
    
    if not wo:
        return None
    
    lots = session.query(WIPRecord).filter(
        WIPRecord.work_order_id == wo.id
    ).all()
    
    summary = {
        'work_order': work_order_number,
        'product': wo.product.part_number,
        'order_quantity': float(wo.order_quantity),
        'lots': [],
        'totals': {
            'total_quantity': 0,
            'good_quantity': 0,
            'scrap_quantity': 0,
            'rework_quantity': 0
        }
    }
    
    for lot in lots:
        lot_info = {
            'lot_number': lot.wip_lot_number,
            'quantity': float(lot.quantity),
            'good': float(lot.good_quantity or 0),
            'scrap': float(lot.scrap_quantity or 0),
            'status': lot.status.value,
            'current_operation': lot.current_operation.operation_code if lot.current_operation else None
        }
        summary['lots'].append(lot_info)
        
        # Update totals
        summary['totals']['total_quantity'] += lot_info['quantity']
        summary['totals']['good_quantity'] += lot_info['good']
        summary['totals']['scrap_quantity'] += lot_info['scrap']
    
    summary['completion_percentage'] = (
        summary['totals']['good_quantity'] / summary['order_quantity'] * 100
        if summary['order_quantity'] > 0 else 0
    )
    
    return summary

# Usage
summary = get_work_order_lot_summary(session, "WO-2024-001")
print(json.dumps(summary, indent=2))

