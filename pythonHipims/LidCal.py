# encoding: utf-8
"""
"""
import sys
import os
import torch
import numpy as np
import sys
import time
import math
import lid
from SWE_CUDA import Godunov
import preProcessing as pre
import postProcessing as post


CASE_PATH = os.path.join(os.environ['HOME'],'LidTest_case')
OUTPUT_PATH = os.path.join(CASE_PATH,'output')
Degree = False
cell_size = 1.0
slope = 0.02
secondOrder = True

def run():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    paraDict = {
        'deviceID': 1,
        'dx': float(cell_size),
        'CFL': 0.5,
        'Export_timeStep': 1.0,
        't': 0.0,
        'export_n': 0,
        'secondOrder': secondOrder,
        'firstTimeStep': 0.01,
        'tensorType': torch.float64,
        'EndTime': 10,
        'Degree': Degree
    }
    dt_list = []
    t_list = []

    # =========================================================
    # set the device
    # ==========================================================
    torch.cuda.set_device(paraDict['deviceID'])
    device = torch.device("cuda", paraDict['deviceID'])

    # =========================================================
    # set tensors
    # ==========================================================
    # prepare the dem

    tensorsize = (int(3 / cell_size), int(3 / cell_size))

    mask = torch.ones(tensorsize, dtype=torch.int32, device=device)
    rigid = 40

    mask[0, :] = rigid
    mask[-1,:] = rigid

    mask[:, 0] = rigid
    mask[:, -1] = 60
    
    # mask[0, :] = -9999
    # mask[-1, :] = -9999
    # mask[:, 0] = -9999
    # mask[:, -1] = -9999

    z = torch.zeros(tensorsize, device=device)
    qx = torch.zeros(tensorsize, device=device)
    qy = torch.zeros(tensorsize, device=device)
    h = torch.zeros(tensorsize, device=device)
    wl = torch.zeros(tensorsize, device=device)

    given_depth = torch.tensor([[0.0, 0.0]],
                                    dtype=paraDict['tensorType'],
                                    device=device)
    given_discharge = torch.tensor([[0.0, 0.0, 0.0]],
                                dtype=paraDict['tensorType'],
                                device=device)

    landuse = torch.zeros(tensorsize, device=device)

    # ====================================================================================================#
    # Test Case 1:only [2,2] lower than central cell
    for i in range(tensorsize[0]):
        for j in range(tensorsize[1]):
            y = (i+0.5) * cell_size
            x = (j+0.5) * cell_size
            if i== 1 and j == 1:
                z[i,j] = 99.85
            elif i== 1 and j == 2:
                z[i,j] = 99.95
            else:
                z[i,j] = 99.99

    # for i in range(tensorsize[0]):
    #     for j in range(tensorsize[1]):
    #         y = (i+0.5) * cell_size
    #         x = (j+0.5) * cell_size            
    #         z[i,j] = 100.0 - x * slope
    #z[1,1] = z[1,2] - 0.1
    h[1,1] = 0.06
    wl = h + z
    Manning = [0.035]
    print(z)

    gauge_index_1D = torch.tensor([])
    gauge_index_1D = gauge_index_1D.to(device)

    rainfallMatrix = np.array([[0., 0.0], [1500, 0.0]])

    # ===============================================
    # set field data
    # ===============================================
    numerical = Godunov(device,
                        paraDict['dx'],
                        paraDict['CFL'],
                        paraDict['Export_timeStep'],
                        t = paraDict['t'],
                        export_n = paraDict['export_n'],
                        firstTimeStep = paraDict['firstTimeStep'],
                        secondOrder = paraDict['secondOrder'],
                        tensorType = paraDict['tensorType'])
    numerical.setOutPutPath(OUTPUT_PATH)
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(Manning, device)
    numerical.set_landuse(mask, landuse, device)

    numerical.set_boundary_tensor(given_depth, given_discharge)

    # ======================================================================
    numerical.set_uniform_rainfall_time_index()
    # ======================================================================
    del landuse, h, qx, qy, wl, z
    torch.cuda.empty_cache()

    
    lid.addlid(3,3)
    
    gauge_dataStoreList = []
    numerical.exportField()
    simulation_start = time.time()

    if paraDict['secondOrder']:
        while numerical.t.item() < paraDict['EndTime']:
            #numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList, n)
            numerical.rungeKutta_update(rainfallMatrix, device)
            numerical.time_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())
            numerical.exportField()
            # print(qx_lidinput)
            # print(qy_lidinput)
            #print(numerical.get_h())
            
    else:
        while numerical.t.item() < paraDict['EndTime']:
            #numerical.observeGauges_write(gauge_index_1D, gauge_dataStoreList, n)
            numerical.addFlux()
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())
            print(numerical.get_qx())
            print(numerical.get_h())

    simulation_end = time.time()
    dt_list.append(simulation_end - simulation_start)
    t_list.append(simulation_end - simulation_start)
    dt_array = np.array(dt_list)
    t_array = np.array(t_list)
    gauge_dataStoreList = np.array(gauge_dataStoreList)

    T = np.column_stack((t_array, dt_array))
    print(OUTPUT_PATH)
    np.savetxt(OUTPUT_PATH + '/t.txt', T)
    np.savetxt(OUTPUT_PATH + '/gauges.txt', gauge_dataStoreList)
    print('Total runtime: ', simulation_end - simulation_start)
    post.exportTxt(mask, mask, OUTPUT_PATH)


if __name__ == "__main__":
    run()