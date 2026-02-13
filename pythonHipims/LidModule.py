import torch
import math
import os
import numpy as np
import lidInfil_new #lidInfil
import lidCal_new #lidCal
import friction_implicit_andUpdate_jh
import timeControl
import pandas as pd
import fluxMask
import fluxCalculation_jh_modified_surface
import fluxCal_2ndOrder_jh_improved
import fluxMask_jjh_modified
import gc
from pythonHipims.postProcessing import *
from pythonHipims.lid_config import *

try:
    from SWE_CUDA import Godunov

except ImportError:
    from . import SWE_CUDA
    from .SWE_CUDA import Godunov


class LidCal(Godunov):
    """
    LID-integrated shallow water model
    
    Key State Variables:
    -------------------
    _cumuSurfaceWaterDepth : torch.Tensor
        Water stored in LID surface layer (NOT free surface water)
        
        Definition by LID type:
        - Green Roof (Type 3): Water within berm (virtual ponding layer)
        - Biocell/Rain Garden (Type 1,2): Water in surface storage layer
        - Permeable Pavement (Type 5): Not used (direct infiltration to pavement)
        
        Note: Separate from h_internal (free flowing water)
        
    _h_internal : torch.Tensor
        Free surface water depth (can flow between cells)
    """
    def __init__(self,
                 device,
                 dx,
                 CFL,
                 Export_timeStep,
                 t=0.0,
                 export_n=0,
                 secondOrder=False,
                 firstTimeStep=1.0e-4,
                 tensorType=torch.double):
        Godunov.__init__(self,
                         device,
                         dx,
                         CFL,
                         Export_timeStep,
                         t=t,
                         export_n=export_n,
                         secondOrder=secondOrder,
                         firstTimeStep=firstTimeStep,
                         tensorType=tensorType)

    def setInPutPath(self, inputpath):
        self._inputpath = inputpath

    def init_lidfield_tenor(self, mask, h, qx, qy, wl, z, device):
        """
        Initialize tensors related to LID modelling
        """
        self._h_internal = torch.as_tensor(h[mask > 0].type(self._tensorType),
                                           device=device)
        
        self._qx_internal = torch.as_tensor(qx[mask > 0].type(self._tensorType),
                                            device=device)
        
        self._qy_internal = torch.as_tensor(qy[mask > 0].type(self._tensorType),
                                            device=device)
        
        self._wl_internal = torch.as_tensor(wl[mask > 0].type(self._tensorType),
                                            device=device)
        
        self._z_internal = torch.as_tensor(z[mask > 0].type(self._tensorType),
                                           device=device)

        self._h_max = torch.as_tensor(h[mask > 0].type(self._tensorType),
                                      device=device)

        self._h_update = torch.zeros_like(self._h_internal,
                                          dtype=self._tensorType,
                                          device=device)

        self._qx_update = torch.zeros_like(self._qx_internal,
                                           dtype=self._tensorType,
                                           device=device)

        self._qy_update = torch.zeros_like(self._qy_internal,
                                           dtype=self._tensorType,
                                           device=device)

        self._z_update = torch.zeros_like(self._z_internal,
                                          dtype=self._tensorType,
                                          device=device)

        del h, qx, qy, wl, z
        torch.cuda.empty_cache()

        # =======================================================================
        # self.__index store the neighbor indexes and the self.__index[0] store the internal cell type or/and index
        # =======================================================================
        index_mask = torch.zeros_like(mask, dtype=torch.int32,
                                      device=device) - 1

        # now index are all -1
        index_mask[mask > 0] = torch.tensor(
            [i for i in range((mask[mask > 0]).size()[0])],
            dtype=torch.int32,
            device=device)

        oppo_direction = torch.tensor([[-1, 1], [1, 0], [1, 1], [-1, 0]],
                                      device=device)
        self._normal = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=self._tensorType,
            device=device)

        self._index = torch.zeros(size=(5, self._h_internal.shape[0]),
                                  dtype=torch.int32,
                                  device=device)
        self._index[0] = mask[mask > 0]
        # print(self._index[0])
        for i in range(4):
            self._index[i + 1] = (index_mask.roll(
                oppo_direction[i][0].item(),
                oppo_direction[i][1].item()))[mask > 0]

        # print(self._index.size())
        self._index = torch.flatten(self._index)
        # print(self._index)

        del index_mask, oppo_direction, mask
        torch.cuda.empty_cache()

    def set_lidlanduse(self, mask, landuseMask, landuse_index, lidMask, areaMask, device): 
        """ revise on 04 dec 2025 for new lid paraeter manager"""
        
        self._landuseMask = torch.as_tensor(landuseMask[mask > 0],
                                        dtype=torch.uint8,
                                        device=device)
        self._lidMask = torch.as_tensor(lidMask[mask > 0],
                                        dtype=torch.uint8,
                                        device=device)
        self._areaMask = torch.as_tensor(areaMask[mask > 0],
                                         dtype=torch.double,
                                         device=device)
        self._landuse_index = torch.as_tensor(landuse_index,
                                              dtype=torch.uint8,
                                              device=device)
        
        n_landuse = len(landuse_index)
        lid_original_ids = self._lid_param_manager.get_lid_original_ids()

        for i, lid_id in enumerate(lid_original_ids):
            self._landuseMask[self._lidMask == lid_id] = n_landuse + i
        self._lidMask = (self._lidMask / 10).int()
        self._lidMask = self._lidMask.type(torch.uint8)

        # Store number of LID types (needed by updat_Manning and CUDA kernels)
        self.n_lid_types = len(lid_original_ids)
        self._lid_para_indices = LidParameterManager.get_lid_array_indices(self)
        
        # Clean up memory
        del mask, landuseMask, landuse_index, lidMask, areaMask, lid_original_ids
        torch.cuda.empty_cache()

        

    # def set_lidlanduse(self, mask, landuseMask, landuse_index, lidMask, areaMask, lidmask_index_class, lidmask_index, device):
    #     """Note: The type of landuse must less than 10"""
    #     self._landuseMask = torch.as_tensor(landuseMask[mask > 0],
    #                                         dtype=torch.uint8,
    #                                         device=device)
    #     self._lidMask = torch.as_tensor(lidMask[mask > 0],
    #                                     dtype=torch.uint8,
    #                                     device=device)
    #     self._areaMask = torch.as_tensor(areaMask[mask > 0],
    #                                      dtype=torch.double,
    #                                      device=device)
    #     self._landuse_index = torch.as_tensor(landuse_index,
    #                                           dtype=torch.uint8,
    #                                           device=device)
    #     self._lidmask_index = torch.as_tensor(lidmask_index,
    #                                           dtype=torch.uint8,
    #                                           device=device)

    #     self._lidmask_index = self._lidmask_index + len(landuse_index)
    #     ## replace the landuse cell with LID measures witht the LID surface use
    #     for i in range(0, len(lidmask_index_class), 1):
    #         self._landuseMask[self._lidMask ==
    #                           lidmask_index_class[i]] = self._lidmask_index[i]
    #     self._lidMask = (self._lidMask/10).int() ## find out the speacific LID and calculation method
    #     self._lidMask = self._lidMask.type(torch.uint8)

    #     del mask, landuseMask, landuse_index, lidMask, lidmask_index, areaMask, lidmask_index_class
    #     torch.cuda.empty_cache()

    # def importLidPara(self, sudsPar_path, device):
    #     df = pd.ExcelFile(sudsPar_path, engine='openpyxl')
    #     surdata = pd.read_excel(df, 'Sur', header=0)
    #     data = np.array(surdata)
    #     # Thickness[0]; Void Fraction[1]; Roughness[2]
    #     self._SurPara = torch.as_tensor(
    #         data, dtype=torch.double, device=device).contiguous()

    #     soildata = pd.read_excel(df, 'Soil', header=0)
    #     data = np.array(soildata)
    #     # Thickness[0]; Porosity(void volume / total volume)[1]; Field Capacity[2]; Initial Moisture (wilting point)[3];
    #     # Saturated Hydraulic Conductivity[4]; slope of log(K) v. moisture content curve[5]; Suction Head[6]
    #     self._SoilPara = torch.as_tensor(
    #         data, dtype=torch.double, device=device).contiguous()

    #     stordata = pd.read_excel(df, 'Stor', header=0)
    #     data = np.array(stordata)
    #     # Thickness[0]; Void Fraction[1]; Saturated Hydraulic Conductivity[2]; Clog Factor[3];
    #     self._StorPara = torch.as_tensor(
    #         data, dtype=torch.double, device=device).contiguous()

    #     pavedata = pd.read_excel(df, 'Pave', header=0)
    #     data = np.array(pavedata)
    #     # Thickness[0]; void volume / total volume[1]; impervious area fraction[2]; permeability[3];
    #     # clogging factor[4]; clogging regeneration interval[5]; degree of clogging regeneration[6]
    #     self._PavePara = torch.as_tensor(
    #         data, dtype=torch.double, device=device).contiguous()

    #     draindata = pd.read_excel(df, 'Drain', header=0)
    #     data = np.array(draindata)
    #     # underdrain flow coeff.[0]; underdrain head exponent[1]; offset height of underdrain[2]; rain barrel drain delay time[3];
    #     # head when drain opens[4]; head when drain closes[5]; curve controlling flow rate[6]
    #     self._DrainPara = torch.as_tensor(
    #         data, dtype=torch.double, device=device).contiguous()

    #     dramatdata = pd.read_excel(df, 'DraMat', header=0)
    #     data = np.array(dramatdata)
    #     #Thickness[0]; Void Fraction[1]; Mannings n for green roof drainage mats[2]; slope/roughness term in Manning equation[3];
    #     self._DraMatPara = torch.as_tensor(
    #         data, dtype=torch.double, device=device).contiguous()

    #     del dramatdata, draindata, pavedata, stordata, soildata, surdata, data
    #     torch.cuda.empty_cache()

    # New LID parameter import function Jinghua 04 Dec 2025
    def importLidPara(self, sudsPar_path, device):
        """
        Load LID parameters using LidParameterManager
        """
        manager = LidParameterManager()
        manager.load_from_excel(sudsPar_path)

        # Generate Cuda tensors for each parameter set
        cuda_arrays = manager.prepare_cuda_arrays(device)
        
        # Assign to module attributes
        self._SurPara = cuda_arrays['SurPara']
        self._SoilPara = cuda_arrays['SoilPara']
        self._StorPara = cuda_arrays['StorPara']
        self._PavePara = cuda_arrays['PavePara']
        self._DrainPara = cuda_arrays['DrainPara']
        self._DraMatPara = cuda_arrays['DraMatPara']
        self._ETPara = cuda_arrays['ETPara']

        self._lid_param_manager = manager
        # ⭐ 添加这个验证块
        print("\n" + "="*80)
        print("🔍 VERIFICATION: LID Parameters Loaded")
        print("="*80)
        drain = self._DrainPara.cpu().numpy()
        print(f"Shape: {drain.shape}")
        print(f"DrainCoeff:  {drain[0, 0]}")
        print(f"DrainExpon:  {drain[1, 0]}")
        print(f"DrainOffset: {drain[2, 0]}")
        print("="*80 + "\n")

        torch.cuda.empty_cache()
        return manager.get_lid_original_ids()
        

    # def updat_Manning(self, manning, device):
    #     manninglid = self._SurPara[2, :]
    #     manninglid = manninglid.unsqueeze(0)
    #     if torch.is_tensor(manning):
    #         self._manning = torch.zeros(
    #             size=(1, manning.numel()+self._lidmask_index.numel()), dtype=torch.double, device=device)
    #         self._manning = torch.cat((manning, manninglid), 1)
    #     else:
    #         manningtensor = torch.tensor([manning],
    #                                      dtype=self._tensorType,
    #                                      device=device)
    #         self._manning = torch.zeros(size=(1, manningtensor.numel(
    #         )+self._lidmask_index.numel()), dtype=torch.double, device=device)
    #         self._manning = torch.cat((manningtensor, manninglid), 1)

    #     del manninglid, manningtensor, manning
    #     torch.cuda.empty_cache()
    def updat_Manning(self, manning, device):
        manninglid = self._SurPara[2, :]
        manninglid = manninglid.unsqueeze(0)
        if torch.is_tensor(manning):
            self._manning = torch.zeros(
                size=(1, manning.numel()+self.n_lid_types), dtype=torch.double, device=device)
            self._manning = torch.cat((manning, manninglid), 1)
        else:
            manningtensor = torch.tensor([manning],
                                         dtype=self._tensorType,
                                         device=device)
            self._manning = torch.zeros(size=(1, manningtensor.numel(
            )+self.n_lid_types), dtype=torch.double, device=device)
            self._manning = torch.cat((manningtensor, manninglid), 1)

        del manninglid, manningtensor, manning
        torch.cuda.empty_cache()

    def lidIMDInitial(self, Stor, soilMoi):
        lidInfil_new.lidIMD_ini_new(
            self._landuseMask, self._soilInfilIMDmax, self._soilIMD,
            self._cumuSoilMoisture, self._cumuStorageWaterDepth,
            Stor, soilMoi)
        # print(self._soilIMD)
        torch.cuda.empty_cache()

    def add_cumulativeSurfaceWaterDepth_Field(self, device):
        gc.collect()
        torch.cuda.empty_cache()
        self._cumuSurfaceWaterDepth = torch.zeros_like(self._h_internal,
                                                       device=device)
        self._cumuSoilMoisture = torch.zeros_like(self._h_internal,
                                                  device=device)
        self._cumuStorageWaterDepth = torch.zeros_like(self._h_internal,
                                                       device=device)
        self._cumuPavementWaterDepth = torch.zeros_like(self._h_internal,
                                                        device=device)
        self._drainrate = torch.zeros_like(self._h_internal,
                                            device=device)

    def ini__infiltrationField_tensor(self, device, init_moisture_lid=None):
        """
        initialise LID infiltration related fields
        2026.01.16 Jinghua Jiang add initial_mode and initial_value for different initial soil moisture setting
        Parameters:
        -----------
        device : torch.device
        initial_mode : str
            - 'relative_saturation': relative saturation of soil layer, defualt mode.
                                        Defined between 0.0 and 1.0 representing
                                        0 represents dry (wilting point) and 1 represents full saturation.
            - 'volumetric'(m3/m3): volumetric water content of soil layer, for example. 0.17. refers to directly measured soil moisture data.
        initial_value : float
        """
        self._f_dt = torch.zeros_like(self._h_internal,
                                      device=device)
        m = self._landuse_index.numel()
        # =====================================================
        self.add_cumulativeSurfaceWaterDepth_Field(device)
        # =====================================================
        # self._initSat = torch.zeros_like(
        #     self._manning, dtype=torch.double, device=device) 
        # self._initSat = torch.ones_like(
        #     self._manning, dtype=torch.double, device=device) * 0.1709 #// 12.12 biovali test*  0.147 //12.15 bio exp
        # =====================================================
        Ks_lid = self._SoilPara[4, :]
        Ks_lid = Ks_lid.unsqueeze(0)
        Ks_else = torch.zeros(size=(1, m), dtype=torch.double, device=device)
        self._soilInfilKs = torch.cat((Ks_else, Ks_lid), 1)
        del Ks_lid, Ks_else
        # =====================================================
        S_lid = self._SoilPara[6, :]
        S_lid = S_lid.unsqueeze(0)
        S_else = torch.zeros(size=(1, m), dtype=torch.double, device=device)
        self._soilInfilS = torch.cat((S_else, S_lid), 1)
        del S_else, S_lid
        # =====================================================
        porosity_lid = self._SoilPara[1, :]
        porosity_lid = porosity_lid.unsqueeze(0)
        porosity_else = torch.zeros(
            size=(1, m), dtype=torch.double, device=device)
        Porosity = torch.cat((porosity_else, porosity_lid), 1)
        del porosity_else, porosity_lid
        # =====================================================
        wiltpoint_lid = self._SoilPara[3, :]
        wiltpoint_lid = wiltpoint_lid.unsqueeze(0)
        wiltpoint_else = torch.zeros(
            size=(1, m), dtype=torch.double, device=device)
        Wiltpoint = torch.cat((wiltpoint_else, wiltpoint_lid), 1)
        del wiltpoint_else, wiltpoint_lid
        # =====================================================
        # modify 2026.01.16 Jinghua Jiang for different initial soil moisture setting
        # calculate IMDmax and soilMoi interms of different initial soil moisture setting
        # =====================================================
        if init_moisture_lid is not None:
            moi_else = torch.zeros(size=(1, m), dtype=torch.double, device=device)  
            if not torch.is_tensor(init_moisture_lid):
                val = torch.tensor(init_moisture_lid, dtype=torch.double, device=device)
                if val.numel() == 1:
                    # 如果是一个数，扩展到 LID 类型数量
                    # self._SoilPara.shape[1] 是 LID 类型数量
                    soilMoi_lid = val.expand(1, self._SoilPara.shape[1])
                else:
                    soilMoi_lid = val.unsqueeze(0)
            else:
                soilMoi_lid = init_moisture_lid.to(device)
                if soilMoi_lid.dim() == 1:
                    soilMoi_lid = soilMoi_lid.unsqueeze(0) # 变成 (1, N)
            soilMoi_else = torch.zeros(size=(1, m), dtype=torch.double, device=device)
            soilMoi = torch.cat((soilMoi_else, soilMoi_lid), 1)
            soilMoi = torch.max(Wiltpoint, torch.min(Porosity, soilMoi))
            self._soilInfilIMDmax = Porosity - soilMoi
            self._initSat = (soilMoi - Wiltpoint) / (Porosity - Wiltpoint + 1e-10)
        else:
            self._initSat = torch.ones_like(
                self._manning, dtype=torch.double, device=device)  # 12.12 biovali test*  0.147 //12.15 bio exp

            unSat = 1.0 - self._initSat
            self._soilInfilIMDmax = (Porosity - Wiltpoint) * unSat
            soilMoi = (Porosity - Wiltpoint) * self._initSat + Wiltpoint
        self._soilInfilIMDmax = torch.max(self._soilInfilIMDmax, torch.zeros_like(self._soilInfilIMDmax))
        # =====================================================
        # self._soilInfilIMDmax = (Porosity - Wiltpoint) * unSat
        self._soilInfilLu = 38.2497 * torch.sqrt(self._soilInfilKs)
        self._soilIMD = torch.ones_like(
            self._h_internal, dtype=torch.double, device=device)
        self._soilFu = torch.zeros_like(
            self._h_internal, dtype=torch.double, device=device)

        self._Sat = torch.zeros_like(
            self._h_internal, dtype=torch.int, device=device)
        # del unSat
        # self._Sat = torch.zeros_like(
        #     self._soilInfilLu, dtype=torch.int, device=device)
        # =====================================================
        # set moistrue limits for soil & storage layers
        pavethickness = self._PavePara[0, :]
        pavethickness = pavethickness.unsqueeze(0)
        pavethickness_else = torch.zeros(
            size=(1, m), dtype=torch.double, device=device)
        Pave = torch.cat((pavethickness_else, pavethickness), 1)
        del pavethickness, pavethickness_else 
        # =====================================================
        storthickness = self._StorPara[0, :]
        storthickness = storthickness.unsqueeze(0)
        storthickness_else = torch.zeros(
            size=(1, m), dtype=torch.double, device=device)
        Stor = torch.cat((storthickness_else, storthickness), 1)
        del storthickness_else, storthickness
        # =====================================================
        drainmat = self._DraMatPara[0, :]
        drainmat = drainmat.unsqueeze(0)
        drainmat_else = torch.zeros(
            size=(1, m), dtype=torch.double, device=device)
        Drainmat = torch.cat((drainmat_else, drainmat), 1)
        del drainmat_else, drainmat
        # =====================================================
        soilMoi = (Porosity - Wiltpoint) * self._initSat + Wiltpoint

        self.lidIMDInitial(Stor, soilMoi)

        self.soilLimMin = torch.as_tensor(Wiltpoint)
        self.soilLimMax = torch.as_tensor(Porosity)
        self.paveLimMax = torch.as_tensor(Pave)
        self.storLimMax = torch.as_tensor(Stor)
        self.storLimMax[Drainmat > 0] = Drainmat[Drainmat > 0]

        del Porosity, Wiltpoint, Pave, Stor, Drainmat
    torch.cuda.empty_cache()

    def Gauges_Rate_cal(self, observe_index, qStoreList, device):
        qq = torch.zeros_like(self._h_internal,device=device)
        qq[observe_index] = torch.sqrt(self._qx_internal[observe_index]*self._qx_internal[observe_index] + self._qy_internal[observe_index]*self._qy_internal[observe_index])
        templistx = []
        templistx += list(qq[observe_index].cpu().numpy())
        # templisty = []
        # templisty += list(self._qy_internal[observe_index].cpu().numpy())
        # templisth=[]
        # templisth += list(self._h_internal[observe_index].cpu().numpy())

        # templistx += list(self._qx_internal[observe_index].cpu().numpy())
        qx_sum= np.sum(templistx)
        # ll = observe_index.size().item()
        # # qx_sum= np.sum(templistx)*ll
        # # templisty = []
        # # templisty += list(self._qy_internal[observe_index].cpu().numpy())
        # # qy_sum= np.sum(templisty)/2.0
        # # q_sum = np.sqrt(qx_sum*qx_sum+qy_sum*qy_sum)
        # templisth=[]
        # templisth += list(self._h_internal[observe_index].cpu().numpy())
        # h_sum= np.sum(templisth)*1.0
        qStoreList.append(qx_sum)
        # hStoreList.append(h_sum)

    def observeGauges_write(self, observe_index, dataStoreList, n):
        # if n == 0:
        #     print(self._z_internal[observe_index.item()])

        if n % 10 == 0:
            templist = []
            templist.append(self.t.item())
            templist += list(self._h_internal[observe_index].cpu().numpy())
            templist += list(self._qx_internal[observe_index].cpu().numpy())
            templist += list(self._qy_internal[observe_index].cpu().numpy())
            #templist += list(self._wl_internal[observe_index].cpu().numpy())
            templist += list(
                self._cumuSurfaceWaterDepth[observe_index].cpu().numpy())
            templist += list(
                self._cumuSoilMoisture[observe_index].cpu().numpy())
            templist += list(
                self._cumuStorageWaterDepth[observe_index].cpu().numpy())
            templist += list(
                self._cumuPavementWaterDepth[observe_index].cpu().numpy())
            dataStoreList.append(templist)

    def addlidFlux(self):
        print(self.dt)
        if self._secondOrder:
            self._wetMask = torch.flatten((self._h_internal > 0.0) | (
                self._landuseMask >= self._landuse_index.numel()).type(torch.bool))
            fluxMask_jjh_modified.update(self._wetMask, self._h_internal,
                                         self._index, self._lidMask, self.t)
            self._wetMask = torch.flatten(self._wetMask.nonzero().type(
                torch.int32))
            # print(self._wetMask) lidFlux.addFlux     
            fluxCal_2ndOrder_jh_improved.addFlux(
                self._wetMask,
                self._h_update,
                self._qx_update,
                self._qy_update,
                self._h_internal,
                self._wl_internal,
                self._z_internal,
                self._qx_internal,
                self._qy_internal,
                self._index,
                self._normal,
                self._given_depth,
                self._given_discharge,
                self.dx,
                self.t,
                self.dt,
            )
            torch.cuda.empty_cache()       
            # lidFlux.addFlux(
            #     self._wetMask,
            #     self._landuseMask,
            #     self._landuse_index,
            #     self._h_update,
            #     self._qx_update,
            #     self._qy_update,
            #     self._h_internal,
            #     self._wl_internal,
            #     self._z_internal,
            #     self._qx_internal,
            #     self._qy_internal,
            #     self._index,
            #     self._normal,
            #     self._given_depth,
            #     self._given_discharge,
            #     self.dx,
            #     self.t,
            #     self.dt,
            # )
            # torch.cuda.empty_cache()
        else:
            self._wetMask = torch.flatten((self._h_internal > 0.0) | (
                self._landuseMask >= self._landuse_index.numel()).type(torch.bool))
            fluxMask_jjh_modified.update(self._wetMask, self._h_internal,
                                         self._index, self._lidMask, self.t)
            self._wetMask = torch.flatten(self._wetMask.nonzero().type(
                torch.int32))
            fluxCalculation_jh_modified_surface.addFlux(
                self._wetMask,
                self._h_update,
                self._qx_update,
                self._qy_update,
                self._h_internal,
                self._wl_internal,
                self._z_internal,
                self._qx_internal,
                self._qy_internal,
                self._index,
                self._normal,
                self._given_depth,
                self._given_discharge,
                self.dx,
                self.t,
                self.dt,
            )
            # lidFlux.addFlux(
            #     # fluxCalculation_1stOrder_Hou.addFlux(
            #     self._wetMask,
            #     self._lidMask,
            #     self._h_update,
            #     self._qx_update,
            #     self._qy_update,
            #     self._h_internal,
            #     self._wl_internal,
            #     self._z_internal,
            #     self._qx_internal,
            #     self._qy_internal,
            #     self._index,
            #     self._normal,
            #     self._given_depth,
            #     self._given_discharge,
            #     self.dx,
            #     self.t,
            #     self.dt,
            # )

            # print(self._h_internal)
            torch.cuda.empty_cache()

    def addLidInfiltrationSource(self, device):
        """
        Calculate LID infiltration using Green-Ampt equation
        
        Note: wetMask is sorted by LID type for better GPU performance
        """
        current_wet_indices = self._wetMask.long()
        lid_indices = (self._lidMask > 0).nonzero(as_tuple=False).flatten()
        combined_indices = torch.cat((current_wet_indices, lid_indices))
        unique_indices = torch.unique(combined_indices)
        calculation_mask = unique_indices
        if calculation_mask.numel() > 0:
            wet_lid_types = self._lidMask[calculation_mask]
            sort_indices = torch.argsort(wet_lid_types, stable=True)
            wetMask_sorted = calculation_mask[sort_indices].int()


        # if self._wetMask.numel() > 0:
        #     # Get LID types for wet cells
        #     wet_lid_types = self._lidMask[self._wetMask.long()]
        #     # Sort indices by LID type (stable sort preserves spatial locality within same type)
        #     sort_indices = torch.argsort(wet_lid_types, stable=True)
        #     wetMask_sorted = self._wetMask[sort_indices]
        # else:
        #     wetMask_sorted = self._wetMask

            n_landuse = self._landuse_index.numel()



            # lidInfil.addLidInfiltrationSource(self._wetMask, self._h_update, self._landuseMask, self._lidMask, self._landuse_index, self._lidmask_index, 
        #                                   self._Sat, self._h_internal, self._f_dt, self._soilInfilKs, self._soilInfilS, self._soilInfilIMDmax,
        #                                   self._soilInfilLu, self._soilFu, self._soilIMD, self._SoilPara, self._SurPara,self._cumuSurfaceWaterDepth, self.dt)
            lidInfil_new.addLidInfiltrationSource_new(wetMask_sorted, self._h_update, self._landuseMask, self._lidMask, n_landuse, 
                                            self._Sat, self._h_internal, self._f_dt, self._soilInfilKs, self._soilInfilS, self._soilInfilIMDmax,
                                            self._soilInfilLu, self._soilFu, self._soilIMD, self._SoilPara, self._SurPara,self._cumuSurfaceWaterDepth, 
                                            self._areaMask, self.dt)
        # print(self._h_internal)
        torch.cuda.empty_cache()

    def LidCalculation(self):
        # if self.n_lid_types > 0:
        #     current_wet_indices = self._wetMask.long()
        #     lid_indices = (self._lidMask > 0).nonzero(as_tuple=False).flatten()
        #     if self._wetMask.numel() > 0:
        #         wet_lid_types = self._lidMask[self._wetMask.long()]
        #         sort_indices = torch.argsort(wet_lid_types, stable=True)
        #         wetMask_sorted = self._wetMask[sort_indices]
        #     else:
        #         wetMask_sorted = self._wetMask
        current_wet_indices = self._wetMask.long()
        lid_indices = (self._lidMask > 0).nonzero(as_tuple=False).flatten()
        combined_indices = torch.cat((current_wet_indices, lid_indices))
        unique_indices = torch.unique(combined_indices)
        calculation_mask = unique_indices
        if calculation_mask.numel() > 0:
            wet_lid_types = self._lidMask[calculation_mask]
            sort_indices = torch.argsort(wet_lid_types, stable=True)
            wetMask_sorted = calculation_mask[sort_indices].int()
        n_landuse = self._landuse_index.numel()
        lidCal_new.addLidcalculation_new(wetMask_sorted, self._h_update, self._landuseMask, self._lidMask,
                                    n_landuse, self._areaMask,
                                    self._h_internal, self._f_dt,
                                    self._SurPara, self._SoilPara, self._StorPara,
                                    self._PavePara,
                                    self._DrainPara, self._DraMatPara,
                                    self.soilLimMin, self.soilLimMax,
                                    self.paveLimMax, self.storLimMax,
                                    self._cumuSurfaceWaterDepth,
                                    self._cumuSoilMoisture,
                                    self._cumuStorageWaterDepth,
                                    self._cumuPavementWaterDepth, 
                                    self._drainrate,
                                    self.dx, self.dx, self.dt)
        # lidCal.addLidcalculation(self._wetMask, self._h_update, self._landuseMask, self._lidMask,
        #                             self._landuse_index, self._lidmask_index, self._areaMask,
        #                             self._h_internal, self._f_dt,
        #                             self._SurPara, self._SoilPara, self._StorPara,
        #                             self._PavePara,
        #                             self._DrainPara, self._DraMatPara,
        #                             self.soilLimMin, self.soilLimMax,
        #                             self.paveLimMax, self.storLimMax,
        #                             self._cumuSurfaceWaterDepth,
        #                             self._cumuSoilMoisture,
        #                             self._cumuStorageWaterDepth,
        #                             self._cumuPavementWaterDepth, 
        #                             self._drainrate,
        #                             self.dx, self.dx, self.dt)
        #print(self._h_update[4383188], self._h_internal[4383188])
        torch.cuda.empty_cache()
        # self._areaMask,

    def time_friction_euler_update_cuda(self, device):
        # limit the time step not bigger than the five times of the older time step
        #+(self._landuseMask >= self._landuse_index.numel()) | (self._lidMask >= 1)
        UPPER = 10.
        time_upper = self.dt * UPPER
        self._wetMask = torch.flatten(
            ((self._h_update.abs() > 0.0) |
             (self._h_internal >= 0.0) | (self._lidMask>= 1)).nonzero()).type(torch.int32)
        # self._wetMask=torch.flatten(
        #     ((self._h_update.abs() > 0.0) +
        #      (self._h_internal >= 0.0)).nonzero()).type(torch.int32)
        friction_implicit_andUpdate_jh.addFriction_eulerUpdate(
            self._wetMask, self._h_update, self._qx_update, self._qy_update,
            self._z_update, self._landuseMask, self._h_internal,
            self._wl_internal, self._qx_internal, self._qy_internal,
            self._z_internal, self._manning, self.dt)
        if (self._h_update<0).nonzero().numel() > 0:
            print((self._h_update<0).nonzero().numel())
        self._h_update[:] = 0.
        self._qx_update[:] = 0.
        self._qy_update[:] = 0.
        self._z_update[:] = 0.
        self._f_dt[:] = 0.
        self._h_internal[self._h_internal<0]=0.0

        # self._cumuSurfaceWaterDepth = self._h_internal.clone()
        self._accelerator_dt = torch.full(self._wetMask.size(),
                                          self._maxTimeStep.item(),
                                          dtype=self._tensorType,
                                          device=device)
        timeControl.updateTimestep(
            self._wetMask,
            self._accelerator_dt,
            self._h_max,
            self._h_internal,
            self._qx_internal,
            self._qy_internal,
            self.dx,
            self.cfl,
            self.t,
            self.dt,
        )

        # if self._h_internal[15468]>0.05:
        #     print(self.t)

        if self._accelerator_dt.size(0) != 0:
            self.dt = torch.min(self._accelerator_dt)
        else:
            # do nothing, keep the last time step
            pass
        # self.dt = min(self.dt, self._maxTimeStep)
        self.dt = min(self.dt, time_upper)
        if (self.dt + self.t).item() >= float(self._export_n +
                                              1) * self.export_timeStep:
            self.dt = (self._export_n + 1) * self.export_timeStep - self.t
            self.exportField()
            # self.exportLid()
            self._export_n += 1
            print("give a output")

        self.t += self.dt
        

    # def exportLid(self):
    #     lidten = torch.unique(self._lidMask[self._lidMask>0])
    #     for i in range (lidten.numel()):
    #         lid = lidten[i].item()
    #         save_Sur = self._cumuSurfaceWaterDepth[self._lidMask==lid]
    #         save_Soi = self._cumuSoilMoisture[self._lidMask==lid]
    #         save_Stor = self._cumuStorageWaterDepth[self._lidMask==lid]
    #         torch.save(
    #         save_Sur,
    #         self._outpath + "/Surlid_" + str(lid) + "_" + str(self.t + self.dt) + ".pt",
    #         )
    #         torch.save(
    #         save_Soi,
    #         self._outpath + "/SoilMoilid_" + str(lid) + "_" + str(self.t + self.dt) + ".pt",
    #         )
    #         torch.save(
    #         save_Stor,
    #         self._outpath + "/Storlid_" + str(lid) + "_" + str(self.t + self.dt) + ".pt",
    #         )

    def exportField(self):
        # we decide to save the data to pt files
        # the .pt file will be processed by numpy later at the postprocessing
        torch.save(
            self._h_internal,
            self._outpath + "/h_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._qx_internal,
            self._outpath + "/qx_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._qy_internal,
            self._outpath + "/qy_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._wl_internal,
            self._outpath + "/wl_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._h_max,
            self._outpath + "/h_max_" + str(self.t + self.dt) + ".pt",
        )
        # torch.save(
        #     self._cumuSurfaceWaterDepth,
        #     self._outpath + "/sur_" + str(self.t + self.dt) + ".pt",
        # )
        torch.save(
            self._cumuSoilMoisture,
            self._outpath + "/soil_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._cumuStorageWaterDepth,
            self._outpath + "/stor_" + str(self.t + self.dt) + ".pt",
        )
        torch.save(
            self._drainrate,
            self._outpath + "/drainrate_" + str(self.t + self.dt) + ".pt",
        )
        # CASE_PATH = os.path.join(os.environ['HOME'], 'LivingDeltas', 'BGI', 'Raingarden')
        # RASTER_PATH = os.path.join(CASE_PATH, 'input')
        # exportRaster_tiff(os.path.join(RASTER_PATH, 'dem_cantho_building_road_river_1.tif'),
        #                     self._outpath)

        # h_out = self._h_internal.cpu()
        # h_save = h_out.numpy()
        # np.savetxt(self._outpath + "/h_" +
        #            str(self.t + self.dt) + '.txt', h_save)
        # qx_out = self._qx_internal.cpu()
        # qx_save = qx_out.numpy()
        # np.savetxt(self._outpath + "/qx_" +
        #            str(self.t + self.dt) + '.txt', qx_save)
        # qy_out = self._qy_internal.cpu()
        # qy_save = qy_out.numpy()
        # np.savetxt(self._outpath + "/qy_" +
        #            str(self.t + self.dt) + '.txt', qy_save)
        # wl_out = self._wl_internal.cpu()
        # wl_save = wl_out.numpy()
        # np.savetxt(self._outpath + "/wl_" +
        #            str(self.t + self.dt) + '.txt', wl_save)
        # cumuSoilMoisture = self._cumuSoilMoisture.cpu()
        # cumuSoilMoisture_save = cumuSoilMoisture.numpy()
        # np.savetxt(self._outpath + "/SoilMoi_" +
        #            str(self.t + self.dt) + '.txt', cumuSoilMoisture_save)
        # cumuStorageWaterDepth = self._cumuStorageWaterDepth.cpu()
        # cumuStorageWaterDepth_save = cumuStorageWaterDepth.numpy()
        # np.savetxt(self._outpath + "/StorWD_" +
        #            str(self.t + self.dt) + '.txt', cumuStorageWaterDepth_save)
