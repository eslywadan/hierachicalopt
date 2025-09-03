
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
    