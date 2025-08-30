"""
Validation Service
Provides validation utilities for manufacturing operations using Little's Law
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ValidationService:
    def __init__(self):
        logger.info("✅ Validation Service initialized")

    def validate_littles_law(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate production parameters using Little's Law: WIP = Throughput × Cycle Time
        """
        wip = request.get('wip')
        throughput = request.get('throughput')
        cycle_time = request.get('cycle_time')
        target_production = request.get('target_production')
        
        validation_result = {
            'is_valid': True,
            'compliance_score': 0.0,
            'expected_values': {},
            'deviations': {},
            'recommendations': [],
            'warnings': [],
            'analysis': {}
        }
        
        try:
            # Case 1: All three parameters provided - check compliance
            if all(param is not None for param in [wip, throughput, cycle_time]):
                expected_wip = throughput * cycle_time
                deviation = abs(wip - expected_wip) / max(wip, expected_wip)
                compliance_score = max(0, 1 - deviation)
                
                validation_result.update({
                    'compliance_score': round(compliance_score, 3),
                    'expected_values': {
                        'wip': round(expected_wip, 2),
                        'actual_wip': round(wip, 2)
                    },
                    'deviations': {
                        'absolute': round(abs(wip - expected_wip), 2),
                        'percentage': round(deviation * 100, 1)
                    }
                })
                
                if deviation > 0.2:  # 20% tolerance
                    validation_result['is_valid'] = False
                    validation_result['warnings'].append(
                        f"High deviation from Little's Law: {deviation*100:.1f}%"
                    )
                
                # Provide recommendations
                if compliance_score < 0.8:
                    if wip > expected_wip:
                        validation_result['recommendations'].append(
                            f"Reduce WIP by {wip - expected_wip:.1f} units or increase throughput to {wip/cycle_time:.1f}"
                        )
                    else:
                        validation_result['recommendations'].append(
                            f"Increase WIP by {expected_wip - wip:.1f} units or reduce cycle time to {wip/throughput:.1f}"
                        )
            
            # Case 2: Two parameters provided - calculate the third
            elif wip is not None and throughput is not None:
                calculated_cycle_time = wip / throughput if throughput > 0 else 0
                validation_result['expected_values']['cycle_time'] = round(calculated_cycle_time, 2)
                validation_result['recommendations'].append(
                    f"Based on WIP={wip} and throughput={throughput}, cycle time should be {calculated_cycle_time:.2f}"
                )
                
            elif wip is not None and cycle_time is not None:
                calculated_throughput = wip / cycle_time if cycle_time > 0 else 0
                validation_result['expected_values']['throughput'] = round(calculated_throughput, 2)
                validation_result['recommendations'].append(
                    f"Based on WIP={wip} and cycle time={cycle_time}, throughput should be {calculated_throughput:.2f}"
                )
                
            elif throughput is not None and cycle_time is not None:
                calculated_wip = throughput * cycle_time
                validation_result['expected_values']['wip'] = round(calculated_wip, 2)
                validation_result['recommendations'].append(
                    f"Based on throughput={throughput} and cycle time={cycle_time}, WIP should be {calculated_wip:.2f}"
                )
            
            # Target production analysis
            if target_production is not None and throughput is not None:
                required_days = target_production / throughput if throughput > 0 else float('inf')
                validation_result['analysis']['production_timeline'] = {
                    'target_production': target_production,
                    'current_throughput': throughput,
                    'days_required': round(required_days, 1) if required_days != float('inf') else 'Impossible'
                }
                
                if required_days > 30:
                    validation_result['warnings'].append(
                        f"Target production requires {required_days:.1f} days at current throughput"
                    )
                    validation_result['recommendations'].append(
                        f"Increase throughput to {target_production/30:.1f} to achieve target in 30 days"
                    )
            
            # General health checks
            self._add_health_checks(validation_result, wip, throughput, cycle_time)
            
            # Calculate overall validation score
            validation_result['overall_score'] = self._calculate_overall_score(validation_result)
            
            logger.info(f"🔍 Validation completed - Score: {validation_result['overall_score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            validation_result.update({
                'is_valid': False,
                'error': str(e),
                'warnings': [f"Validation failed: {str(e)}"]
            })
        
        return validation_result

    def validate_production_feasibility(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if production parameters are feasible
        """
        feasibility_result = {
            'is_feasible': True,
            'feasibility_score': 1.0,
            'constraints': [],
            'bottlenecks': [],
            'optimization_suggestions': []
        }
        
        try:
            wip = parameters.get('wip', 0)
            throughput = parameters.get('throughput', 0)
            cycle_time = parameters.get('cycle_time', 0)
            capacity = parameters.get('capacity', float('inf'))
            
            # Check physical constraints
            if wip < 0 or throughput < 0 or cycle_time < 0:
                feasibility_result['is_feasible'] = False
                feasibility_result['constraints'].append("Negative values are not physically possible")
            
            # Check capacity constraints
            if throughput > capacity:
                feasibility_result['is_feasible'] = False
                feasibility_result['constraints'].append(f"Throughput {throughput} exceeds capacity {capacity}")
                feasibility_result['bottlenecks'].append("Capacity bottleneck")
            
            # Check reasonable bounds
            if cycle_time > 30:  # Very long cycle time
                feasibility_result['feasibility_score'] *= 0.7
                feasibility_result['constraints'].append("Cycle time unusually high")
                feasibility_result['optimization_suggestions'].append("Consider process improvements to reduce cycle time")
            
            if cycle_time < 0.1:  # Very short cycle time
                feasibility_result['feasibility_score'] *= 0.8
                feasibility_result['constraints'].append("Cycle time unusually low - may be unrealistic")
            
            # WIP reasonableness
            if wip > 1000:  # Very high WIP
                feasibility_result['feasibility_score'] *= 0.8
                feasibility_result['optimization_suggestions'].append("High WIP detected - consider lean improvements")
            
            # Throughput efficiency
            if capacity != float('inf'):
                efficiency = throughput / capacity
                if efficiency < 0.3:
                    feasibility_result['optimization_suggestions'].append(
                        f"Low capacity utilization ({efficiency*100:.1f}%) - consider increasing throughput"
                    )
                elif efficiency > 0.9:
                    feasibility_result['optimization_suggestions'].append(
                        f"High capacity utilization ({efficiency*100:.1f}%) - monitor for bottlenecks"
                    )
            
            logger.info(f"🏭 Feasibility check completed - Score: {feasibility_result['feasibility_score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Feasibility validation error: {str(e)}")
            feasibility_result.update({
                'is_feasible': False,
                'error': str(e)
            })
        
        return feasibility_result

    def _add_health_checks(self, validation_result: Dict, wip: float, throughput: float, cycle_time: float):
        """Add general health checks to validation result"""
        if wip is not None and wip <= 0:
            validation_result['warnings'].append("WIP should be positive")
        
        if throughput is not None and throughput <= 0:
            validation_result['warnings'].append("Throughput should be positive")
        
        if cycle_time is not None and cycle_time <= 0:
            validation_result['warnings'].append("Cycle time should be positive")
        
        # Check for unreasonable values
        if cycle_time is not None and cycle_time > 20:
            validation_result['warnings'].append("Cycle time seems unusually high")
        
        if wip is not None and wip > 500:
            validation_result['warnings'].append("WIP level seems unusually high")

    def _calculate_overall_score(self, validation_result: Dict) -> float:
        """Calculate overall validation score"""
        score = validation_result.get('compliance_score', 0.5)
        
        # Penalize for warnings
        warning_count = len(validation_result.get('warnings', []))
        score *= max(0.3, 1 - warning_count * 0.1)
        
        # Penalize if not valid
        if not validation_result.get('is_valid', True):
            score *= 0.5
        
        return round(score, 3)

    def generate_optimization_recommendations(self, current_state: Dict[str, Any], 
                                           target_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate optimization recommendations to move from current to target state
        """
        recommendations = {
            'priority_actions': [],
            'parameter_adjustments': {},
            'expected_improvements': {},
            'implementation_steps': []
        }
        
        try:
            current_wip = current_state.get('wip', 0)
            current_throughput = current_state.get('throughput', 0)
            current_cycle_time = current_state.get('cycle_time', 0)
            
            target_wip = target_state.get('wip')
            target_throughput = target_state.get('throughput')
            target_cycle_time = target_state.get('cycle_time')
            
            # Analyze gaps
            if target_throughput and target_throughput > current_throughput:
                throughput_gap = target_throughput - current_throughput
                recommendations['priority_actions'].append(
                    f"Increase throughput by {throughput_gap:.1f} units"
                )
                
                # Suggest specific actions
                recommendations['implementation_steps'].extend([
                    "1. Analyze current bottlenecks",
                    "2. Optimize equipment utilization",
                    "3. Consider parallel processing",
                    "4. Improve process efficiency"
                ])
            
            if target_cycle_time and target_cycle_time < current_cycle_time:
                cycle_time_reduction = current_cycle_time - target_cycle_time
                recommendations['priority_actions'].append(
                    f"Reduce cycle time by {cycle_time_reduction:.1f}"
                )
                
                recommendations['implementation_steps'].extend([
                    "1. Identify process waste",
                    "2. Implement lean methodologies",
                    "3. Automate manual processes",
                    "4. Optimize workflow"
                ])
            
            if target_wip and target_wip < current_wip:
                wip_reduction = current_wip - target_wip
                recommendations['priority_actions'].append(
                    f"Reduce WIP by {wip_reduction:.1f} units"
                )
                
                recommendations['implementation_steps'].extend([
                    "1. Implement pull system",
                    "2. Balance production line",
                    "3. Reduce batch sizes",
                    "4. Improve material flow"
                ])
            
            logger.info("🎯 Optimization recommendations generated")
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            recommendations['error'] = str(e)
        
        return recommendations