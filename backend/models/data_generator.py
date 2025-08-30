"""
Data Generator Service
Generates synthetic manufacturing data for LSTM model training
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
import math

logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, master_data_service=None):
        self.master_data_service = master_data_service
        logger.info("📊 Data Generator Service initialized")

    def generate_data(self, config: Dict, params: Dict) -> List[Dict[str, Any]]:
        """
        Generate synthetic manufacturing data based on configuration
        """
        logger.info(f"🏭 DataGenerator.generate_data called!")
        logger.info(f"🏭 Config received: {config}")
        logger.info(f"🏭 Params received: {params}")
        logger.info(f"🏭 Generating synthetic data for {config.get('historical_days', 0)} days")
        
        # Use synchronized master data if available and populated
        if self.master_data_service:
            sync_config = self.master_data_service.get_synchronized_config()
            sync_plants = sync_config.get('plants', [])
            sync_applications = sync_config.get('applications', [])
            sync_panel_sizes = sync_config.get('panel_sizes', [])
            
            # Only use synchronized data if it's actually populated
            if sync_plants and sync_applications and sync_panel_sizes:
                plants = sync_plants
                applications = sync_applications
                panel_sizes = sync_panel_sizes
                logger.info(f"🔄 Using synchronized master data: {len(plants)} plants, {len(applications)} applications, {len(panel_sizes)} panel sizes")
            else:
                plants = config['plants']
                applications = config['applications']
                panel_sizes = config['panel_sizes']
                logger.info("⚠️ Master data service available but no data loaded, using provided config data")
            
        else:
            plants = config['plants']
            applications = config['applications']
            panel_sizes = config['panel_sizes']
            logger.info("⚠️ Using provided config data (master data service not available)")
        
        data = []
        start_date = datetime.now() - timedelta(days=config['historical_days'])
        
        # Base parameters
        base_wip = params['base_wip']
        base_throughput = params['base_throughput']
        seasonality = params['seasonality']
        noise_level = params['noise_level']
        
        for day in range(config['historical_days']):
            current_date = start_date + timedelta(days=day)
            
            # Seasonal factor (yearly cycle)
            seasonal_factor = 1 + seasonality * math.sin(2 * math.pi * day / 365)
            
            # Weekly pattern (lower on weekends)
            weekly_factor = 0.7 if current_date.weekday() >= 5 else 1.0
            
            for plant in plants:
                plant_factor = self._get_plant_factor(plant)
                
                for application in applications:
                    app_factor = self._get_application_factor(application)
                    
                    for panel_size in panel_sizes:
                        size_factor = self._get_size_factor(panel_size)
                        
                        # Generate correlated WIP and throughput with independent variations
                        base_wip_adj = base_wip * plant_factor * app_factor * size_factor
                        base_throughput_adj = base_throughput * plant_factor * app_factor * size_factor
                        
                        # Independent random variations for WIP and throughput
                        wip_noise = np.random.normal(0, noise_level)
                        throughput_noise = np.random.normal(0, noise_level)
                        
                        wip = base_wip_adj * seasonal_factor * weekly_factor * (1 + wip_noise)
                        throughput = base_throughput_adj * seasonal_factor * weekly_factor * (1 + throughput_noise)
                        
                        # Ensure positive values
                        wip = max(1, wip)
                        throughput = max(0.1, throughput)
                        
                        # Calculate cycle time from Little's Law with some variation
                        cycle_time = wip / throughput
                        
                        # Add some realistic constraints
                        cycle_time = max(0.5, min(cycle_time, 20))  # Reasonable cycle time bounds
                        
                        # Calculate production metrics
                        finished_goods = throughput * 0.8 + np.random.normal(0, throughput * 0.1)
                        semi_finished_goods = wip * 0.3 + np.random.normal(0, wip * 0.05)
                        
                        # Ensure non-negative values
                        finished_goods = max(0, finished_goods)
                        semi_finished_goods = max(0, semi_finished_goods)
                        
                        # Calculate Little's Law compliance
                        expected_wip = throughput * cycle_time
                        littles_law_compliance = 1 - abs(wip - expected_wip) / max(wip, expected_wip)
                        littles_law_compliance = max(0, min(1, littles_law_compliance))
                        
                        data_point = {
                            'day': day,
                            'date': current_date.isoformat(),
                            'plant': plant,
                            'application': application,
                            'panel_size': panel_size,
                            'wip': round(wip, 2),
                            'throughput': round(throughput, 2),
                            'cycle_time': round(cycle_time, 2),
                            'finished_goods': round(finished_goods, 2),
                            'semi_finished_goods': round(semi_finished_goods, 2),
                            'littles_law_compliance': round(littles_law_compliance, 3)
                        }
                        
                        data.append(data_point)
        
        logger.info(f"✅ Generated {len(data)} data points")
        
        # Log some statistics if data exists
        if data:
            try:
                df = pd.DataFrame(data)
                stats = {
                    'avg_wip': df['wip'].mean(),
                    'avg_throughput': df['throughput'].mean(),
                    'avg_cycle_time': df['cycle_time'].mean(),
                    'avg_littles_compliance': df['littles_law_compliance'].mean()
                }
                
                logger.info(f"📊 Data statistics: {stats}")
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate statistics: {str(e)}")
                logger.info(f"Sample data point: {data[0] if data else 'No data'}")
        else:
            logger.error("❌ No data generated!")
        
        return data

    def _get_plant_factor(self, plant: str) -> float:
        """Get plant-specific scaling factor"""
        plant_factors = {
            # Old names (backward compatibility)
            'Plant_A': 1.0,
            'Plant_B': 1.2,
            'Plant_C': 0.8,
            # New real plant names
            'Taiwan_Fab1': 1.0,
            'China_Fab1': 1.2,
            'Korea_Fab1': 0.8,
            'PLT_TW01': 1.0,
            'PLT_CN01': 1.2,
            'PLT_KR01': 0.8
        }
        return plant_factors.get(plant, 1.0)

    def _get_application_factor(self, application: str) -> float:
        """Get application-specific scaling factor"""
        app_factors = {
            # Old names (backward compatibility)
            'Automotive': 1.3,
            'Consumer_Electronics': 1.0,
            'Industrial': 1.5,
            # New real application names
            'Commercial Display': 1.1,
            'Consumer TV': 1.0,
            'Gaming Monitor': 1.3,
            'Laptop Display': 1.2,
            'Professional Monitor': 1.4
        }
        return app_factors.get(application, 1.0)

    def _get_size_factor(self, panel_size: str) -> float:
        """Get panel size-specific scaling factor"""
        size_factors = {
            # Old names (backward compatibility)
            'Small': 1.5,
            'Medium': 1.0,
            'Large': 0.8,
            'Extra_Large': 0.6,
            # New real panel size names
            '15.6"': 1.5,
            '21.5"': 1.3,
            '27"': 1.0,
            '32"': 0.9,
            '43"': 0.8,
            '55"': 0.7,
            '65"': 0.6
        }
        return size_factors.get(panel_size, 1.0)

    def generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate predefined test scenarios for model validation"""
        scenarios = [
            {
                'name': 'High Volume - Small Panels',
                'plant': 'Plant_A',
                'application': 'Consumer_Electronics',
                'panel_size': 'Small',
                'current_wip': 150,
                'planned_throughput': 80,
                'target_production': 2000
            },
            {
                'name': 'Low Volume - Large Panels',
                'plant': 'Plant_B',
                'application': 'Automotive',
                'panel_size': 'Large',
                'current_wip': 80,
                'planned_throughput': 20,
                'target_production': 500
            },
            {
                'name': 'Industrial - Medium Panels',
                'plant': 'Plant_C',
                'application': 'Industrial',
                'panel_size': 'Medium',
                'current_wip': 120,
                'planned_throughput': 45,
                'target_production': 1200
            }
        ]
        
        logger.info(f"🧪 Generated {len(scenarios)} test scenarios")
        
        return scenarios

    def validate_data_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of generated data"""
        df = pd.DataFrame(data)
        
        validation_results = {
            'total_points': len(data),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'feature_stats': {
                'wip': {
                    'min': float(df['wip'].min()),
                    'max': float(df['wip'].max()),
                    'mean': float(df['wip'].mean()),
                    'std': float(df['wip'].std())
                },
                'throughput': {
                    'min': float(df['throughput'].min()),
                    'max': float(df['throughput'].max()),
                    'mean': float(df['throughput'].mean()),
                    'std': float(df['throughput'].std())
                },
                'cycle_time': {
                    'min': float(df['cycle_time'].min()),
                    'max': float(df['cycle_time'].max()),
                    'mean': float(df['cycle_time'].mean()),
                    'std': float(df['cycle_time'].std())
                }
            },
            'quality_checks': {
                'negative_values': int(((df[['wip', 'throughput', 'cycle_time']] < 0).any(axis=1)).sum()),
                'zero_values': int(((df[['wip', 'throughput', 'cycle_time']] == 0).any(axis=1)).sum()),
                'outliers': int((df['cycle_time'] > 20).sum() + (df['cycle_time'] < 0.1).sum()),
                'littles_law_compliance': float(df['littles_law_compliance'].mean())
            },
            'coverage': {
                'plants': list(df['plant'].unique()),
                'applications': list(df['application'].unique()),
                'panel_sizes': list(df['panel_size'].unique())
            }
        }
        
        # Determine data quality score
        quality_score = 1.0
        if validation_results['quality_checks']['negative_values'] > 0:
            quality_score -= 0.3
        if validation_results['quality_checks']['zero_values'] > 0:
            quality_score -= 0.2
        if validation_results['quality_checks']['outliers'] > len(data) * 0.05:
            quality_score -= 0.2
        if validation_results['quality_checks']['littles_law_compliance'] < 0.7:
            quality_score -= 0.3
        
        validation_results['quality_score'] = max(0, quality_score)
        
        logger.info(f"📋 Data quality validation complete - Score: {quality_score:.2f}")
        
        return validation_results