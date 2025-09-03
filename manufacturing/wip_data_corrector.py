from manufacturing.wip_track import WIPHistory
from zoneinfo import ZoneInfo
from datetime import datetime

local_tz = ZoneInfo("Asia/Taipei")

class WIPDataCorrector:
    """Handle backward corrections for WIP history data"""
    
    @staticmethod
    def correct_good_quantity(session, wip_lot_number, correct_good_qty, 
                            event_timestamp=None, create_audit=True):
        """
        Correct good_quantity for a specific lot
        
        Args:
            session: Database session
            wip_lot_number: Lot number to correct
            correct_good_qty: The correct good quantity value
            event_timestamp: Specific event to correct (if None, corrects all)
            create_audit: Whether to create an audit trail
        """
        
        # Find the affected records
        query = session.query(WIPHistory).filter(
            WIPHistory.wip_lot_number == wip_lot_number
        )
        
        if event_timestamp:
            query = query.filter(WIPHistory.event_timestamp == event_timestamp)
        
        records = query.all()
        
        corrections = []
        for record in records:
            if record.good_quantity and record.good_quantity > record.quantity:
                old_value = record.good_quantity
                
                # Apply correction logic
                if correct_good_qty is not None:
                    record.good_quantity = min(correct_good_qty, record.quantity)
                else:
                    # Auto-correct to 90% of quantity if no value provided
                    record.good_quantity = record.quantity * 0.9
                
                correction_info = {
                    'record_id': record.id,
                    'lot_number': record.wip_lot_number,
                    'event_timestamp': record.event_timestamp,
                    'old_good_qty': float(old_value),
                    'new_good_qty': float(record.good_quantity),
                    'corrected_at': datetime.now(local_tz)
                }
                corrections.append(correction_info)
                
                # Create audit entry
                if create_audit:
                    audit_note = (f"Data correction: good_quantity changed from "
                                f"{old_value} to {record.good_quantity}")
                    
                    audit_history = WIPHistory(
                        wip_record_id=record.wip_record_id,
                        wip_lot_number=record.wip_lot_number,
                        work_order_number=record.work_order_number,
                        event_type='data_correction',
                        event_timestamp=datetime.now(local_tz),
                        operation_id=record.operation_id,
                        operation_code=record.operation_code,
                        quantity=record.quantity,
                        good_quantity=record.good_quantity,
                        notes=audit_note,
                        created_by='data_correction_system'
                    )
                    session.add(audit_history)
        
        session.commit()
        return corrections
    
    @staticmethod
    def bulk_correct_by_ratio(session, max_yield_percentage=98.0):
        """
        Bulk correct all records where good quantity exceeds reasonable yield
        
        Args:
            session: Database session
            max_yield_percentage: Maximum reasonable yield percentage
        """
        
        problematic_records = session.query(WIPHistory).filter(
            WIPHistory.good_quantity > WIPHistory.quantity * (max_yield_percentage / 100)
        ).all()
        
        corrections = []
        for record in problematic_records:
            old_good = record.good_quantity
            # Cap at maximum yield
            record.good_quantity = min(
                record.good_quantity,
                record.quantity * (max_yield_percentage / 100)
            )
            
            corrections.append({
                'lot': record.wip_lot_number,
                'old': float(old_good),
                'new': float(record.good_quantity)
            })
        
        session.commit()
        print(f"Corrected {len(corrections)} records")
        return corrections

    @staticmethod
    def create_correction_report(session, wip_lot_number):
        """Generate a detailed correction report for a lot"""
        
        history = session.query(WIPHistory).filter(
            WIPHistory.wip_lot_number == wip_lot_number
        ).order_by(WIPHistory.event_timestamp).all()
        
        report = {
            'lot_number': wip_lot_number,
            'issues_found': [],
            'suggested_corrections': []
        }
        
        for record in history:
            if record.good_quantity and record.quantity:
                yield_pct = (record.good_quantity / record.quantity) * 100
                
                if yield_pct > 100:
                    issue = {
                        'event_timestamp': record.event_timestamp,
                        'event_type': record.event_type,
                        'quantity': float(record.quantity),
                        'good_quantity': float(record.good_quantity),
                        'yield_percentage': yield_pct,
                        'issue': 'Yield exceeds 100%'
                    }
                    report['issues_found'].append(issue)
                    
                    # Suggest correction
                    from decimal import Decimal
                    suggested_good = record.quantity * Decimal('0.9')  # Assume 90% yield
                    report['suggested_corrections'].append({
                        'record_id': record.id,
                        'suggested_good_quantity': suggested_good,
                        'reasoning': 'Applied standard 90% yield'
                    })
        
        return report
    

