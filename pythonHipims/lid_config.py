#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: lid_config.py
Description: Configuration and parameter management for HiPIMS-LID module.
"""

import torch
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


# =============================================================================
# 1. Parameter Dataclasses
# =============================================================================

@dataclass
class LidSurfaceParams:
    """Surface layer parameters"""
    thickness: float  # m
    void_fraction: float 
    roughness: float  # Manning n
    
    def to_tensor(self, device):
        return torch.tensor([self.thickness, self.void_fraction, self.roughness], 
                          dtype=torch.float64, device=device)


@dataclass
class LidSoilParams:
    """Soil layer parameters"""
    thickness: float  # m
    porosity: float 
    field_capacity: float 
    wilting_point: float 
    saturated_conductivity: float  # m/s
    conductivity_slope: float  # slope of K(θ) curve
    suction_head: float  # suction head (m)
    
    def to_tensor(self, device):
        return torch.tensor([
            self.thickness, self.porosity, self.field_capacity,
            self.wilting_point, self.saturated_conductivity,
            self.conductivity_slope, self.suction_head
        ], dtype=torch.float64, device=device)
    
    @property
    def available_porosity(self):
        return self.porosity - self.field_capacity


@dataclass
class LidStorageParams:
    """Storage layer parameters"""
    thickness: float  # m
    void_fraction: float 
    saturated_conductivity: float  # m/s
    clog_factor: float 
    
    def to_tensor(self, device):
        return torch.tensor([
            self.thickness, self.void_fraction,
            self.saturated_conductivity, self.clog_factor
        ], dtype=torch.float64, device=device)


@dataclass
class LidPavementParams:
    """Pavement layer parameters"""
    thickness: float  # m
    void_fraction: float 
    impervious_fraction: float 
    permeability: float  # m/s
    clog_factor: float 
    
    def to_tensor(self, device):
        return torch.tensor([
            self.thickness, self.void_fraction, self.impervious_fraction,
            self.permeability, self.clog_factor
        ], dtype=torch.float64, device=device)


@dataclass
class LidDrainParams:
    """Drainage system parameters"""
    coefficient: float  # C
    exponent: float  # n
    offset: float  # m
    
    def to_tensor(self, device):
        return torch.tensor([self.coefficient, self.exponent, self.offset],
                          dtype=torch.float64, device=device)


@dataclass
class LidDrainageMatParams:
    """Drainage mat parameters (e.g. for Green Roof)"""
    thickness: float  # m
    void_fraction: float 
    manning_coefficient: float 
    alpha: float 
    
    def to_tensor(self, device):
        return torch.tensor([self.thickness, self.void_fraction, self.alpha],
                          dtype=torch.float64, device=device)


@dataclass
class LidETParams:
    """Evapotranspiration parameters (Simplified for Hydrodynamic Model)"""
    constant_rate: float = 0.0  # mm/day (or mm/hr, depending on your model units)
    surface_fraction: float = 0.0  # fraction of surface available for evaporation
    
    def to_tensor(self, device):
        """
        Convert ET params to tensor.
        Tensor layout (Dim=2): [constant_rate, surface_fraction]
        """
        # 直接返回两个参数，不再需要 method code
        return torch.tensor([self.constant_rate, self.surface_fraction],
                          dtype=torch.float64, device=device)

    def get_rate_at_time(self, t: float) -> float:
        return self.constant_rate

class LidTypeConfig:
    """Complete configuration for a single LID type"""
    
    def __init__(self, lid_id: int, lid_type: int):
        self.lid_id  = lid_id         # Original ID (10, 11, 20, 30, ...)
        self.lid_type  = lid_type   # Type code (1, 2, 3, ...) for CUDA branching
        
        # Parameters for each layer
        self.surface: Optional[LidSurfaceParams] = None
        self.soil: Optional[LidSoilParams] = None
        self.storage: Optional[LidStorageParams] = None
        self.pavement: Optional[LidPavementParams] = None
        self.drain: Optional[LidDrainParams] = None
        self.drainage_mat: Optional[LidDrainageMatParams] = None
        self.et: Optional[LidETParams] = None
    
    
    def to_tensor_dict(self, device) -> Dict[str, torch.Tensor]:
        """Convert to dictionary of tensors (useful for debugging or single-instance use)"""
        tensors = {}
        if self.surface: tensors['surface'] = self.surface.to_tensor(device)
        if self.soil: tensors['soil'] = self.soil.to_tensor(device)
        if self.storage: tensors['storage'] = self.storage.to_tensor(device)
        if self.pavement: tensors['pavement'] = self.pavement.to_tensor(device)
        if self.drain: tensors['drain'] = self.drain.to_tensor(device)
        if self.drainage_mat: tensors['drainage_mat'] = self.drainage_mat.to_tensor(device)
        if self.et: tensors['et'] = self.et.to_tensor(device)
        return tensors


# =============================================================================
# 2. State Variable Management
# =============================================================================

class LidStateArrays:
    """Manage all LID state variables in unified tensors"""
    
    def __init__(self, n_cells: int, device, dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        
        # Layer depths / moisture
        self.surface_depth = torch.zeros(n_cells, dtype=dtype, device=device)
        self.soil_moisture = torch.zeros(n_cells, dtype=dtype, device=device)  # θ
        self.storage_depth = torch.zeros(n_cells, dtype=dtype, device=device)
        self.pavement_depth = torch.zeros(n_cells, dtype=dtype, device=device)
        
        # Green-Ampt specific
        self.Fu = torch.zeros(n_cells, dtype=dtype, device=device)  # Upper zone moisture volume
        self.F = torch.zeros(n_cells, dtype=dtype, device=device)   # Cumulative infiltration volume
        self.IMD = torch.zeros(n_cells, dtype=dtype, device=device) # Initial Moisture Deficit
        self.Sat = torch.zeros(n_cells, dtype=torch.bool, device=device) # Saturation flag
        
        # Cumulative trackers
        self.cumulative_infiltration = torch.zeros(n_cells, dtype=dtype, device=device)
        self.cumulative_ET = torch.zeros(n_cells, dtype=dtype, device=device)
        self.cumulative_drainage = torch.zeros(n_cells, dtype=dtype, device=device)
    
    def sync_greenampt_with_soil(self, lid_params: LidSoilParams):
        """Sync Green-Ampt variables (Fu) based on current Soil Moisture (θ)"""
        # Fu = (θ - field_capacity) * thickness
        # Clamp at 0 because θ cannot be less than WP effectively for gravity drainage logic
        self.Fu = torch.clamp(
            (self.soil_moisture - lid_params.field_capacity) * lid_params.thickness,
            min=0.0
        )
        # IMD = (porosity - θ) * thickness (This is actually Deficit relative to Saturation)
        self.IMD = (lid_params.porosity - self.soil_moisture) * lid_params.thickness
    
    def sync_soil_with_greenampt(self, lid_params: LidSoilParams):
        """Sync Soil Moisture (θ) based on current Green-Ampt Fu"""
        # θ = field_capacity + Fu / thickness
        self.soil_moisture = lid_params.field_capacity + self.Fu / lid_params.thickness
        
        # Enforce physical bounds
        self.soil_moisture = torch.clamp(
            self.soil_moisture,
            min=lid_params.wilting_point,
            max=lid_params.porosity
        )
    
    def check_consistency(self, lid_params: LidSoilParams, tolerance=1e-6):
        """Check consistency between Fu and Soil Moisture"""
        expected_Fu = (self.soil_moisture - lid_params.field_capacity) * lid_params.thickness
        expected_Fu = torch.clamp(expected_Fu, min=0.0)
        
        diff = torch.abs(self.Fu - expected_Fu)
        max_diff = diff.max().item()
        
        if max_diff > tolerance:
            print(f"WARNING: State inconsistency detected. Max diff = {max_diff:.6e}")
            return False
        return True


# =============================================================================
# 3. Parameter Manager
# =============================================================================

class LidParameterManager:
    """
    Loads and manages LID parameters for HiPIMS-LID integration
    
    Key concepts:
    - lid_configs: Dict[lid_id, config] where lid_id is original ID (10, 20, 30, ...)
    - lid_id_to_index: Maps original LID_ID to array index (0, 1, 2, ...)
    - CUDA arrays use remapped indices for memory efficiency
    """
    
    def __init__(self):
        self.lid_configs: Dict[int, LidTypeConfig] = {}
        self.lid_id_to_index: Dict[int, int] = {}  # LID_ID to array index
        self.index_to_lid_id: Dict[int, int] = {}  # # array index to LID_ID
        self.n_lid_types = 0 # Number of loaded LID types
    
    def load_from_excel(self, excel_path: str):
        """
        Load LID configurations from Excel file
        
        Excel format:
        - LID_ID: Original ID (10, 11, 20, 30, ...) - REQUIRED
        - LID_Type: Type code (1, 2, 3, ...) 
        - Name: Human-readable name - OPTIONAL (ignored)
        - Sur_Thick, Soil_Thick, etc.: Parameter columns
        
        Parameters:
        -----------
        excel_path: str
            Path to Excel file with LID parameters
        """
        # Read Excel, fill NaN with 0 to prevent errors
        df = pd.read_excel(excel_path, header=0).fillna(0)
        df = df.sort_values(by='LID_ID').reset_index(drop=True)
        
        for remapped_idx, (_, row) in enumerate(df.iterrows()):
            lid_id = int(row['LID_ID'])
            lid_type = int(row['LID_Type'])

            # Update ID mappings
            self.lid_id_to_index[lid_id] = remapped_idx
            self.index_to_lid_id[remapped_idx] = lid_id
            
            config = LidTypeConfig(lid_id, lid_type)
            
            # --- Surface Layer ---
            config.surface = LidSurfaceParams(
                thickness=float(row['Sur_Thick']),
                void_fraction=float(row['Sur_Void']),
                roughness=float(row['Sur_Rough'])
            )
            
            # --- Soil Layer ---
            config.soil = LidSoilParams(
                thickness=float(row['Soil_Thick']),
                porosity=float(row['Soil_Por']),
                field_capacity=float(row['Soil_FC']),
                wilting_point=float(row['Soil_WP']),
                saturated_conductivity=float(row['Soil_Ks']),
                conductivity_slope=float(row['Soil_Kslope']),
                suction_head=float(row['Soil_Suction'])
            )
            
            # --- Storage Layer ---
            config.storage = LidStorageParams(
                thickness=float(row['Stor_Thick']),
                void_fraction=float(row['Stor_Void']),
                saturated_conductivity=float(row['Stor_Ks']),
                clog_factor=float(row['Stor_Clog'])
            )
            
            # --- Pavement Layer (Optional) ---
            if row['Pave_Thick'] > 0:
                config.pavement = LidPavementParams(
                    thickness=float(row['Pave_Thick']),
                    void_fraction=float(row['Pave_Void']),
                    impervious_fraction=float(row['Pave_Frac']),
                    permeability=float(row['Pave_Perm']),
                    clog_factor=float(row['Pave_Clog'])
                )
            
            # --- Drain System ---
            config.drain = LidDrainParams(
                coefficient=float(row['Drain_Coeff']),
                exponent=float(row['Drain_Exp']),
                offset=float(row['Drain_Off'])
            )
            
            # --- Drainage Mat (Optional) ---
            if row['Mat_Thick'] > 0:
                config.drainage_mat = LidDrainageMatParams(
                    thickness=float(row['Mat_Thick']),
                    void_fraction=float(row['Mat_Void']),
                    manning_coefficient=float(row['Mat_Manning']),
                    alpha=float(row['Mat_Alpha'])
                )

            # --- ET (Optional) ---
            # Using .get to avoid error if column doesn't exist yet
            et_method = str(row.get('ET_Method', '0')).strip()
            if et_method and et_method not in ['0', '0.0', 'nan', '']:
                config.et = LidETParams(
                    constant_rate=float(row.get('ET_Const', 0.0)),
                    surface_fraction=float(row.get('ET_SurfFrac', 0.0))
                )
            
            self.lid_configs[lid_id] = config
        self.n_lid_types = len(self.lid_configs)

        # Print summary
        print(f"\n✅ Successfully loaded {self.n_lid_types} LID configurations")
        print(f"   LID_IDs: {sorted(self.lid_configs.keys())}")
        print(f"\n   Index mapping (for CUDA arrays):")
        for lid_id in sorted(self.lid_configs.keys()):
            idx = self.lid_id_to_index[lid_id]
            lid_type = self.lid_configs[lid_id].lid_type
            print(f"      LID_ID {lid_id:2d} (Type {lid_type}) → array index {idx}")
    
    def get_config(self, lid_id: int) -> Optional[LidTypeConfig]:
        """Get configuration by LID_ID"""
        return self.lid_configs.get(lid_id)
    
    def get_remapped_index(self, lid_id: int) -> Optional[int]:
        """Get array index for a given LID_ID"""
        return self.lid_id_to_index.get(lid_id)
    
    def prepare_cuda_arrays(self, device) -> Dict[str, torch.Tensor]:
        """
        Generate CUDA parameter arrays
        
        Returns:
        --------
        Dict[str, torch.Tensor]
            Parameter arrays with shape [N_params, n_lid_types]
            - Column index corresponds to remapped index (0, 1, 2, ...)
            - NOT to original LID_ID
        """
        num_lid_types = self.n_lid_types
        print(f"Preparing CUDA arrays for {num_lid_types} LID types")
        # 1. Define dimensions
        dims = {
            'Sur': 3, 'Soil': 7, 'Stor': 4, 'Pave': 5, 'Drain': 3, 'DraMat': 3, 'ET': 2
        }
        
        # 2. Initialize tensors with zeros [Param_Count, Lid_Type_Count]
        arrays = {
            'SurPara': torch.zeros((dims['Sur'], num_lid_types), dtype=torch.float64, device=device),
            'SoilPara': torch.zeros((dims['Soil'], num_lid_types), dtype=torch.float64, device=device),
            'StorPara': torch.zeros((dims['Stor'], num_lid_types), dtype=torch.float64, device=device),
            'PavePara': torch.zeros((dims['Pave'], num_lid_types), dtype=torch.float64, device=device),
            'DrainPara': torch.zeros((dims['Drain'], num_lid_types), dtype=torch.float64, device=device),
            'DraMatPara': torch.zeros((dims['DraMat'], num_lid_types), dtype=torch.float64, device=device),
            'ETPara': torch.zeros((dims['ET'], num_lid_types), dtype=torch.float64, device=device)
        }

        # 3. Populate tensors
        for lid_id, config in self.lid_configs.items():
            remapped_idx = self.lid_id_to_index[lid_id]
            
            if config.surface:
                arrays['SurPara'][:, remapped_idx] = config.surface.to_tensor(device)
            if config.soil:
                arrays['SoilPara'][:, remapped_idx] = config.soil.to_tensor(device)
            if config.storage:
                arrays['StorPara'][:, remapped_idx] = config.storage.to_tensor(device)
            if config.pavement:
                arrays['PavePara'][:, remapped_idx] = config.pavement.to_tensor(device)
            if config.drain:
                arrays['DrainPara'][:, remapped_idx] = config.drain.to_tensor(device)
            if config.drainage_mat:
                arrays['DraMatPara'][:, remapped_idx] = config.drainage_mat.to_tensor(device)
            if config.et:
                arrays['ETPara'][:, remapped_idx] = config.et.to_tensor(device)
        
        # 4. Ensure contiguous memory
        for k in arrays:
            arrays[k] = arrays[k].contiguous()
                    
        return arrays

    def get_lid_original_ids(self) -> np.ndarray:
        """
        Get array of original LID_IDs (sorted order)
        Used for set_lidlanduse function
        
        Returns:
        --------
        np.ndarray
            Original LID_IDs in sorted order, e.g. [10, 20, 30]
        """
        sorted_ids = sorted(self.lid_configs.keys())
        return np.array(sorted_ids, dtype=np.int32)
    
    def get_lid_array_indices(self) -> np.ndarray:
        """
        Get array of remapped indices [0, 1, 2, ...]
        Used for set_lidlanduse function
        
        Returns:
        --------
        np.ndarray
            Remapped indices, e.g. [0, 1, 2]
        """
        return np.arange(self.n_lid_types, dtype=np.int32)
    

# =============================================================================
# 4. Example Usage
# =============================================================================

def example_usage():
    """Demonstrate usage of LidParameterManager"""
    print("\n" + "="*70)
    print("LidParameterManager Usage Example")
    print("="*70)
    
    # 1. Load parameters
    manager = LidParameterManager()
    manager.load_from_excel('/home/lunet/cvjj7/DARe_NBS/Para_new.xlsx')
    
    # 2. Check all loaded configs
    print("\n📋 Loaded configurations:")
    for lid_id in sorted(manager.lid_configs.keys()):
        config = manager.get_config(lid_id)
        idx = manager.get_remapped_index(lid_id)
        print(f"   LID {lid_id} (Type {config.lid_type}) → index {idx}")
        print(f"      Soil: {config.soil.thickness}m, Por={config.soil.porosity}")
    
    # 3. Generate CUDA arrays
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cuda_arrays = manager.prepare_cuda_arrays(device)
    
    # 5. Verify array contents
    print(f"\n✅ Verification - checking all LID parameters:")
    for lid_id in sorted(manager.lid_configs.keys()):
        idx = manager.get_remapped_index(lid_id)
        print(f"\n   LID {lid_id} → array index {idx}:")
        print(f"      SoilPara[:, {idx}] = {cuda_arrays['SoilPara'][:, idx].cpu().numpy()}")
        print(f"      SurPara[:, {idx}]  = {cuda_arrays['SurPara'][:, idx].cpu().numpy()}")


if __name__ == "__main__":
    example_usage()