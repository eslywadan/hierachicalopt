"""
Master Data Management Service
Manages synchronized master data across the hierarchical optimization system
"""
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Set
import os
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class PlantMaster:
    """Master data for plants"""
    plant_id: str
    plant_name: str
    display_name: str
    location: str
    region: str
    capacity_per_day: int
    specializations: List[str]
    panel_sizes: List[str]
    technology_generation: str
    automation_level: str

@dataclass 
class ProductMaster:
    """Master data for products/applications"""
    product_id: str
    product_name: str
    display_name: str
    category: str
    panel_sizes: List[str]
    market_segments: List[str]

@dataclass
class PanelSizeMaster:
    """Master data for panel sizes"""
    size_id: str
    size_display: str
    diagonal_inches: float
    category: str  # Small, Medium, Large, Extra_Large
    applications: List[str]

@dataclass
class RegionMaster:
    """Master data for regions (used by both demand and plant data)"""
    region_id: str
    region_name: str
    display_name: str

@dataclass
class DataConsistencyReport:
    """Data consistency validation report"""
    timestamp: datetime
    total_issues: int
    plant_issues: List[str]
    product_issues: List[str]
    panel_size_issues: List[str]
    demand_data_issues: List[str]
    plant_data_issues: List[str]
    recommendations: List[str]
    is_consistent: bool

class MasterDataService:
    def __init__(self):
        """Initialize Master Data Service"""
        self.plants: Dict[str, PlantMaster] = {}
        self.products: Dict[str, ProductMaster] = {}
        self.panel_sizes: Dict[str, PanelSizeMaster] = {}
        self.regions: Dict[str, RegionMaster] = {}
        self.data_file_paths = {
            'plant_data': '../src/assets/data/tft_lcd_plant_data.csv',
            'demand_data': '../src/assets/data/tft_lcd_demand_data.csv',
            'component_data': '../src/assets/data/tft_lcd_component_data.csv',
            'market_data': '../src/assets/data/tft_lcd_market_data.csv'
        }
        logger.info("🗃️ Master Data Service initialized")
        
        # Load master data on initialization
        self.load_master_data()
    
    def load_master_data(self):
        """Load master data from CSV files"""
        try:
            # Load plant master data
            self._load_plant_master_data()
            
            # Load product master data from demand patterns
            self._load_product_master_data()
            
            # Load panel size master data
            self._load_panel_size_master_data()
            
            # Load region master data
            self._load_region_master_data()
            
            logger.info(f"✅ Master data loaded: {len(self.plants)} plants, {len(self.products)} products, {len(self.panel_sizes)} panel sizes, {len(self.regions)} regions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load master data: {str(e)}")
            # Initialize with fallback data
            self._initialize_fallback_master_data()
    
    def _load_plant_master_data(self):
        """Load plant master data from plant CSV"""
        plant_file = self.data_file_paths['plant_data']
        
        if os.path.exists(plant_file):
            df = pd.read_csv(plant_file)
            
            for _, row in df.iterrows():
                # Extract panel sizes from capabilities
                panel_sizes = []
                for col in df.columns:
                    if col.startswith('Can_Produce_') and row[col]:
                        size = col.replace('Can_Produce_', '').replace('in', '"')
                        panel_sizes.append(size)
                
                # Extract specializations
                specializations = []
                if pd.notna(row.get('Specialization_1')):
                    specializations.append(row['Specialization_1'])
                if pd.notna(row.get('Specialization_2')):
                    specializations.append(row['Specialization_2'])
                
                plant = PlantMaster(
                    plant_id=row['Plant_ID'],
                    plant_name=row['Plant_Name'],
                    display_name=row['Plant_Name'].replace('_', ' '),
                    location=row['Location'],
                    region=row['Region'],
                    capacity_per_day=int(row['Capacity_Per_Day']),
                    specializations=specializations,
                    panel_sizes=panel_sizes,
                    technology_generation=row.get('Technology_Generation', 'Unknown'),
                    automation_level=row.get('Automation_Level', 'Medium')
                )
                
                self.plants[plant.plant_id] = plant
                # Also index by plant name for lookup
                self.plants[plant.plant_name] = plant
        else:
            logger.warning(f"Plant data file not found: {plant_file}")
    
    def _load_product_master_data(self):
        """Load product master data from demand patterns"""
        demand_file = self.data_file_paths['demand_data']
        
        if os.path.exists(demand_file):
            df = pd.read_csv(demand_file)
            
            # Extract unique products/market segments
            market_segments = df['Market_Segment'].unique()
            
            for segment in market_segments:
                segment_data = df[df['Market_Segment'] == segment]
                panel_sizes = segment_data['Panel_Size'].unique().tolist()
                
                # Map market segments to product categories
                category_mapping = {
                    'Commercial Display': 'commercial',
                    'Consumer TV': 'consumer',
                    'Gaming Monitor': 'gaming', 
                    'Laptop Display': 'laptop',
                    'Professional Monitor': 'professional'
                }
                
                product = ProductMaster(
                    product_id=f"PRD_{segment.replace(' ', '_').upper()}",
                    product_name=segment,
                    display_name=segment,
                    category=category_mapping.get(segment, 'general'),
                    panel_sizes=panel_sizes,
                    market_segments=[segment]
                )
                
                self.products[product.product_id] = product
                self.products[product.product_name] = product
        else:
            logger.warning(f"Demand data file not found: {demand_file}")
    
    def _load_panel_size_master_data(self):
        """Load panel size master data"""
        demand_file = self.data_file_paths['demand_data']
        
        if os.path.exists(demand_file):
            df = pd.read_csv(demand_file)
            
            # Extract unique panel sizes and their applications
            panel_sizes = df['Panel_Size'].unique()
            
            for size in panel_sizes:
                size_data = df[df['Panel_Size'] == size]
                applications = size_data['Market_Segment'].unique().tolist()
                
                # Parse diagonal inches
                try:
                    diagonal = float(size.replace('"', ''))
                except:
                    diagonal = 0.0
                
                # Categorize by size
                if diagonal <= 21.5:
                    category = 'Small'
                elif diagonal <= 32:
                    category = 'Medium'
                elif diagonal <= 55:
                    category = 'Large'
                else:
                    category = 'Extra_Large'
                
                panel = PanelSizeMaster(
                    size_id=f"SIZE_{size.replace('\"', '_IN')}",
                    size_display=size,
                    diagonal_inches=diagonal,
                    category=category,
                    applications=applications
                )
                
                self.panel_sizes[panel.size_id] = panel
                self.panel_sizes[panel.size_display] = panel
        else:
            logger.warning(f"Demand data file not found: {demand_file}")
    
    def _load_region_master_data(self):
        """Load unified region master data from both demand and plant data"""
        regions_set = set()
        
        # Get regions from demand data (customer regions)
        demand_file = self.data_file_paths['demand_data']
        if os.path.exists(demand_file):
            df = pd.read_csv(demand_file)
            demand_regions = df['Region'].unique()
            regions_set.update(demand_regions)
        
        # Get regions from plant data (plant regions)
        plant_file = self.data_file_paths['plant_data']
        if os.path.exists(plant_file):
            df = pd.read_csv(plant_file)
            plant_regions = df['Region'].unique()
            regions_set.update(plant_regions)
        
        # Create region master data
        for region_name in regions_set:
            region = RegionMaster(
                region_id=f"RGN_{region_name.replace(' ', '_').upper()}",
                region_name=region_name,
                display_name=region_name
            )
            self.regions[region.region_id] = region
            self.regions[region.region_name] = region
    
    def _initialize_fallback_master_data(self):
        """Initialize fallback master data when CSV loading fails"""
        logger.info("🔄 Initializing fallback master data")
        
        # Fallback plants based on common naming patterns
        fallback_plants = [
            PlantMaster("PLT_TW01", "Taiwan_Fab1", "Taiwan Fab 1", "Taiwan", "Asia Pacific", 5000, 
                       ["Consumer TV", "Gaming Monitor"], ['27"', '32"', '43"', '55"', '65"'], "Gen 10.5", "High"),
            PlantMaster("PLT_CN01", "China_Fab1", "China Fab 1", "Shanghai", "Asia Pacific", 8000,
                       ["Consumer TV", "Commercial Display"], ['21.5"', '27"', '32"', '43"', '55"', '65"'], "Gen 8.5", "Medium"),
            PlantMaster("PLT_KR01", "Korea_Fab1", "Korea Fab 1", "Seoul", "Asia Pacific", 4000,
                       ["Gaming Monitor", "Professional Monitor"], ['15.6"', '21.5"', '27"', '32"'], "Gen 8.5", "High")
        ]
        
        for plant in fallback_plants:
            self.plants[plant.plant_id] = plant
            self.plants[plant.plant_name] = plant
        
        # Fallback products
        fallback_products = [
            ProductMaster("PRD_COMMERCIAL_DISPLAY", "Commercial Display", "Commercial Display", "commercial",
                         ['21.5"', '32"', '43"', '55"'], ["Commercial Display"]),
            ProductMaster("PRD_CONSUMER_TV", "Consumer TV", "Consumer TV", "consumer",
                         ['32"', '43"', '55"', '65"'], ["Consumer TV"]),
            ProductMaster("PRD_GAMING_MONITOR", "Gaming Monitor", "Gaming Monitor", "gaming",
                         ['21.5"', '27"', '32"'], ["Gaming Monitor"]),
            ProductMaster("PRD_LAPTOP_DISPLAY", "Laptop Display", "Laptop Display", "laptop",
                         ['15.6"'], ["Laptop Display"]),
            ProductMaster("PRD_PROFESSIONAL_MONITOR", "Professional Monitor", "Professional Monitor", "professional",
                         ['21.5"', '27"', '32"'], ["Professional Monitor"])
        ]
        
        for product in fallback_products:
            self.products[product.product_id] = product
            self.products[product.product_name] = product
        
        # Fallback panel sizes
        fallback_panel_sizes = [
            PanelSizeMaster("SIZE_15_6_IN", '15.6"', 15.6, "Small", ["Laptop Display"]),
            PanelSizeMaster("SIZE_21_5_IN", '21.5"', 21.5, "Small", ["Commercial Display", "Gaming Monitor", "Professional Monitor"]),
            PanelSizeMaster("SIZE_27_IN", '27"', 27.0, "Medium", ["Gaming Monitor", "Professional Monitor"]),
            PanelSizeMaster("SIZE_32_IN", '32"', 32.0, "Medium", ["Commercial Display", "Consumer TV", "Gaming Monitor", "Professional Monitor"]),
            PanelSizeMaster("SIZE_43_IN", '43"', 43.0, "Large", ["Commercial Display", "Consumer TV"]),
            PanelSizeMaster("SIZE_55_IN", '55"', 55.0, "Large", ["Commercial Display", "Consumer TV"]),
            PanelSizeMaster("SIZE_65_IN", '65"', 65.0, "Extra_Large", ["Consumer TV"])
        ]
        
        for panel in fallback_panel_sizes:
            self.panel_sizes[panel.size_id] = panel
            self.panel_sizes[panel.size_display] = panel
        
        # Fallback regions (unified for both customer and plant regions)
        fallback_regions = [
            RegionMaster("RGN_ASIA_PACIFIC", "Asia Pacific", "Asia Pacific"),
            RegionMaster("RGN_NORTH_AMERICA", "North America", "North America"),
            RegionMaster("RGN_EUROPE", "Europe", "Europe"),
            RegionMaster("RGN_LATIN_AMERICA", "Latin America", "Latin America")
        ]
        
        for region in fallback_regions:
            self.regions[region.region_id] = region
            self.regions[region.region_name] = region
    
    def get_synchronized_config(self) -> Dict[str, Any]:
        """Get synchronized configuration for data generation"""
        return {
            'plants': [plant.plant_name for plant in self.plants.values() if hasattr(plant, 'plant_name')],
            'applications': [product.product_name for product in self.products.values() if hasattr(product, 'product_name')],
            'panel_sizes': [panel.size_display for panel in self.panel_sizes.values() if hasattr(panel, 'size_display')],
            'regions': [region.region_name for region in self.regions.values() if hasattr(region, 'region_name')]
        }
    
    def validate_data_consistency(self) -> DataConsistencyReport:
        """Validate data consistency across all data sources"""
        logger.info("🔍 Starting data consistency validation")
        
        issues = {
            'plant_issues': [],
            'product_issues': [],
            'panel_size_issues': [],
            'demand_data_issues': [],
            'plant_data_issues': []
        }
        recommendations = []
        
        try:
            # Check plant data consistency
            self._validate_plant_consistency(issues)
            
            # Check product data consistency 
            self._validate_product_consistency(issues)
            
            # Check panel size consistency
            self._validate_panel_size_consistency(issues)
            
            # Check demand data alignment
            self._validate_demand_data_alignment(issues)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues)
            
        except Exception as e:
            logger.error(f"❌ Error during data consistency validation: {str(e)}")
            issues['plant_issues'].append(f"Validation error: {str(e)}")
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        report = DataConsistencyReport(
            timestamp=datetime.now(),
            total_issues=total_issues,
            plant_issues=issues['plant_issues'],
            product_issues=issues['product_issues'],
            panel_size_issues=issues['panel_size_issues'],
            demand_data_issues=issues['demand_data_issues'],
            plant_data_issues=issues['plant_data_issues'],
            recommendations=recommendations,
            is_consistent=total_issues == 0
        )
        
        logger.info(f"📋 Consistency validation complete: {total_issues} issues found")
        return report
    
    def _validate_plant_consistency(self, issues: Dict):
        """Validate plant data consistency"""
        plant_file = self.data_file_paths['plant_data']
        if os.path.exists(plant_file):
            df = pd.read_csv(plant_file)
            
            # Check if plant regions exist in region master data
            plant_regions = df['Region'].unique()
            master_regions = set(region.region_name for region in self.regions.values() if hasattr(region, 'region_name'))
            
            for region in plant_regions:
                if region not in master_regions:
                    issues['plant_issues'].append(f"Plant data references region '{region}' not found in region master data")
            
            # Check for missing required plant data columns
            required_columns = ['Plant_ID', 'Plant_Name', 'Location', 'Region', 'Capacity_Per_Day']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            for col in missing_columns:
                issues['plant_data_issues'].append(f"Missing required column in plant data: {col}")
    
    def _validate_product_consistency(self, issues: Dict):
        """Validate product data consistency"""
        demand_file = self.data_file_paths['demand_data']
        if os.path.exists(demand_file):
            df = pd.read_csv(demand_file)
            demand_segments = df['Market_Segment'].unique()
            
            master_products = set(product.product_name for product in self.products.values() if hasattr(product, 'product_name'))
            
            for segment in demand_segments:
                if segment not in master_products:
                    issues['product_issues'].append(f"Demand data references market segment '{segment}' not found in product master")
            
            # Check if demand regions exist in region master data
            demand_regions = df['Region'].unique()
            master_regions = set(region.region_name for region in self.regions.values() if hasattr(region, 'region_name'))
            
            for region in demand_regions:
                if region not in master_regions:
                    issues['product_issues'].append(f"Demand data references region '{region}' not found in region master data")
    
    def _validate_panel_size_consistency(self, issues: Dict):
        """Validate panel size consistency"""
        demand_file = self.data_file_paths['demand_data']
        if os.path.exists(demand_file):
            df = pd.read_csv(demand_file)
            demand_sizes = df['Panel_Size'].unique()
            
            master_sizes = set(panel.size_display for panel in self.panel_sizes.values() if hasattr(panel, 'size_display'))
            
            for size in demand_sizes:
                if size not in master_sizes:
                    issues['panel_size_issues'].append(f"Demand data references panel size '{size}' not found in panel size master")
        
        # Cross-validate panel sizes in plant data
        plant_file = self.data_file_paths['plant_data']
        if os.path.exists(plant_file):
            df = pd.read_csv(plant_file)
            
            # Check panel size capability columns
            panel_capability_columns = [col for col in df.columns if col.startswith('Can_Produce_')]
            
            for col in panel_capability_columns:
                # Handle both formats: 15_6in -> 15.6" and 21_5in -> 21.5"
                size_raw = col.replace('Can_Produce_', '').replace('in', '')
                if '_' in size_raw:
                    size = size_raw.replace('_', '.') + '"'
                else:
                    size = size_raw + '"'
                
                master_sizes = set(panel.size_display for panel in self.panel_sizes.values() if hasattr(panel, 'size_display'))
                
                if size not in master_sizes:
                    issues['panel_size_issues'].append(f"Plant data references panel size capability '{size}' not found in panel size master")
    
    def _validate_demand_data_alignment(self, issues: Dict):
        """Validate demand data internal consistency"""
        demand_file = self.data_file_paths['demand_data']
        if os.path.exists(demand_file):
            df = pd.read_csv(demand_file)
            
            # Check for missing required columns
            required_columns = ['Date', 'Region', 'Panel_Size', 'Market_Segment', 'Forecasted_Demand', 'Actual_Demand']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            for col in missing_columns:
                issues['demand_data_issues'].append(f"Missing required column: {col}")
            
            # Check for null values in critical fields
            for col in required_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        issues['demand_data_issues'].append(f"Column '{col}' has {null_count} null values")
    
    def _generate_recommendations(self, issues: Dict) -> List[str]:
        """Generate recommendations based on found issues"""
        recommendations = []
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        if total_issues == 0:
            recommendations.append("✅ All data is consistent! No action required.")
            return recommendations
        
        if issues['plant_issues']:
            recommendations.append("🏭 Update plant master data to include all regions referenced in demand data")
            recommendations.append("🔄 Consider creating a region mapping table for better data alignment")
        
        if issues['product_issues']:
            recommendations.append("📦 Synchronize product master data with market segments in demand data")
            recommendations.append("🏷️ Create standardized product naming conventions")
        
        if issues['panel_size_issues']:
            recommendations.append("📏 Ensure all panel sizes in demand data have corresponding master data entries")
            recommendations.append("🔧 Standardize panel size format across all data sources")
        
        if issues['demand_data_issues']:
            recommendations.append("🗃️ Clean demand data to remove null values in critical columns")
            recommendations.append("📊 Implement data quality checks for demand data ingestion")
        
        recommendations.append("🔄 Run this consistency check regularly to maintain data quality")
        recommendations.append("📋 Consider implementing automated data validation in the data pipeline")
        
        return recommendations
    
    def get_master_data_summary(self) -> Dict[str, Any]:
        """Get summary of master data for dashboard display"""
        return {
            'plants': {
                'count': len(set(plant.plant_id for plant in self.plants.values() if hasattr(plant, 'plant_id'))),
                'regions': list(set(plant.region for plant in self.plants.values() if hasattr(plant, 'region'))),
                'items': [
                    {
                        'id': plant.plant_id,
                        'name': plant.plant_name,
                        'display_name': plant.display_name,
                        'location': plant.location,
                        'region': plant.region,
                        'capacity': plant.capacity_per_day,
                        'specializations': plant.specializations,
                        'panel_sizes': plant.panel_sizes
                    }
                    for plant in self.plants.values() 
                    if hasattr(plant, 'plant_id')
                ][:10]  # Limit for display
            },
            'products': {
                'count': len(set(product.product_id for product in self.products.values() if hasattr(product, 'product_id'))),
                'categories': list(set(product.category for product in self.products.values() if hasattr(product, 'category'))),
                'items': [
                    {
                        'id': product.product_id,
                        'name': product.product_name,
                        'display_name': product.display_name,
                        'category': product.category,
                        'panel_sizes': product.panel_sizes
                    }
                    for product in self.products.values()
                    if hasattr(product, 'product_id')
                ]
            },
            'panel_sizes': {
                'count': len(set(panel.size_id for panel in self.panel_sizes.values() if hasattr(panel, 'size_id'))),
                'categories': list(set(panel.category for panel in self.panel_sizes.values() if hasattr(panel, 'category'))),
                'items': [
                    {
                        'id': panel.size_id,
                        'display': panel.size_display,
                        'diagonal': panel.diagonal_inches,
                        'category': panel.category,
                        'applications': panel.applications
                    }
                    for panel in self.panel_sizes.values()
                    if hasattr(panel, 'size_id')
                ]
            },
            'regions': {
                'count': len(set(region.region_id for region in self.regions.values() if hasattr(region, 'region_id'))),
                'items': [
                    {
                        'id': region.region_id,
                        'name': region.region_name,
                        'display_name': region.display_name
                    }
                    for region in self.regions.values()
                    if hasattr(region, 'region_id')
                ]
            }
        }
    
    def export_master_data_config(self) -> str:
        """Export master data configuration as JSON"""
        config = self.get_synchronized_config()
        summary = self.get_master_data_summary()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'synchronized_config': config,
            'master_data_summary': summary,
            'version': '1.0'
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)