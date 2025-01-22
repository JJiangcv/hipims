# encoding: utf-8
"""
"""
import os
import torch
import numpy as np
import time
from hipims.pythonHipims.LidModule import LidCal

try:
    import postProcessing as post
    import preProcessing as pre
    # import LidModule as LidCal
except ImportError:
    from . import postProcessing as post
    from . import preProcessing as pre
    # from . import LidModule as LidCal


def run(paraDict):

    # ===============================================
    # Make output folder
    # ===============================================
    dt_list = []
    t_list = []
    if not os.path.isdir(paraDict['OUTPUT_PATH']):
        os.mkdir(paraDict['OUTPUT_PATH'])

    # ===============================================
    # set the device
    # ===============================================
    torch.cuda.set_device(paraDict['deviceID'])
    device = torch.device("cuda", paraDict['deviceID'])

    # ===============================================
    # set the tensors
    # ===============================================
    pre.setTheMainDevice(device)
    dem, mask, demMeta, gauge_index_1D = pre.importDEMData_And_BC(
        paraDict['rasterPath']['DEM_path'],
        device,
        gauges_position=paraDict['gauges_position'], default_BC=paraDict['default_BC'],
        boundBox=paraDict['boundBox'],
        bc_type=paraDict['bc_type'])
    if paraDict['Degree']:
        paraDict['dx'] = pre.degreeToMeter(demMeta['transform'][0])
    else:
        paraDict['dx'] = demMeta['transform'][0]

    landuse, landuse_index = pre.importLanduseData(
        paraDict['rasterPath']['Landuse_path'], device, paraDict['landLevel'])
    # rainfall_station_Mask = pre.importRainStationMask(
    #     paraDict['rasterPath']['Rainfall_path'], device)
    lidmask, lidmask_index_class, lidmask_index = pre.importLidData(
        paraDict['rasterPath']['LidMask_path'], device)

    areamask = pre.importAreaMask(
        paraDict['rasterPath']['areaMask_path'], device)

    gauge_index_1D = gauge_index_1D.to(device)
    z = dem
    h = torch.zeros_like(z, device=device)
    qx = torch.zeros_like(z, device=device)
    qy = torch.zeros_like(z, device=device)
    wl = h + z

    #manning = np.array([0.035, 0.1, 0.035, 0.04, 0.15, 0.03])
    # ===============================================
    # rainfall data
    # ===============================================
    # rainfallMatrix = pre.voronoiDiagramGauge_rainfall_source(
    #     paraDict['Rainfall_data_Path'])

    # if "climateEffect" in paraDict:
    #     rainfallMatrix[:, 1:] *= paraDict['climateEffect']

    # ===============================================
    # set field data
    # ===============================================
    numerical = LidCal(device,
                       paraDict['dx'],
                       paraDict['CFL'],
                       paraDict['Export_timeStep'],
                       t=paraDict['t'],
                       export_n=paraDict['export_n'],
                       firstTimeStep=paraDict['firstTimeStep'],
                       secondOrder=paraDict['secondOrder'],
                       tensorType=paraDict['tensorType'])

    numerical.setOutPutPath(paraDict['OUTPUT_PATH'])
    numerical.init__fluidField_tensor(mask, h, qx, qy, wl, z, device)
    numerical.set__frictionField_tensor(paraDict['Manning'], device)
    numerical.set_lidlanduse(mask, landuse, landuse_index, lidmask, areamask,
                             lidmask_index_class, lidmask_index, device)
    numerical.importLidPara(paraDict['rasterPath']['SudsPara_path'], device)
    #lidnum = numerical.init_LidPara_tensor(landuse_index, device)

    numerical.updat_Manning(paraDict['Manning'], device)

    numerical.ini__infiltrationField_tensor(device)
    # numerical.set_boundary_tensor(paraDict['given_h'], paraDict['given_q'])
    # ======================================================================
    # #numerical.set_distributed_rainfall_station_Mask(mask,
    #                                                 rainfall_station_Mask,
    #                                                 device)
    # ======================================================================
    # uniform rainfall test
    # ======================================================================
    rainfallMatrix = np.array([[0., 3.0e-06], [3600.00, 3.0e-06],
                               [5400.00, 3.0e-06], [5401.00, 0.0],
                               [7200.00, 0.0], [12010.00, 0.0]])
    numerical.set_uniform_rainfall_time_index()
    # ======================================================================

    del mask, landuse, h, qx, qy, wl, z
    torch.cuda.empty_cache()
    numerical.exportField()
    simulation_start = time.time()

    gauge_dataStoreList = []
    # lid_dataStoreList = []
    n = 0
    if gauge_index_1D.size()[0] > 0:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.observeGauges_write(
                gauge_index_1D, gauge_dataStoreList, 0)
            # numerical.rungeKutta_update(rainfallMatrix, device)
            numerical.addFlux()
            numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
            hlid = numerical.get_h()
            numerical.addLidInfiltrationSource(device)
            hlid = numerical.get_h()
            numerical.LidCalculation()
            numerical.time_friction_euler_update_cuda(device)

            # numerical.time_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            n += 1
            print(numerical.t.item())
    else:
        while numerical.t.item() < paraDict['EndTime']:
            numerical.addFlux()
            numerical.add_uniform_PrecipitationSource(rainfallMatrix, device)
            numerical.addLidInfiltrationSource()
            numerical.LidCalculation()
            numerical.time_friction_euler_update_cuda(device)
            dt_list.append(numerical.dt.item())
            t_list.append(numerical.t.item())
            print(numerical.t.item())

    simulation_end = time.time()
    dt_list.append(simulation_end - simulation_start)
    t_list.append(simulation_end - simulation_start)
    dt_array = np.array(dt_list)
    t_array = np.array(t_list)
    gauge_dataStoreList = np.array(gauge_dataStoreList)

    T = np.column_stack((t_array, dt_array))
    np.savetxt(paraDict['OUTPUT_PATH'] + '/t.txt', T)
    np.savetxt(paraDict['OUTPUT_PATH'] + '/gauges.txt', gauge_dataStoreList)
    # post.exportRaster_tiff(paraDict['rasterPath']['DEM_path'],
    #                        paraDict['OUTPUT_PATH'])
    post.cutSectPlot(paraDict['rasterPath']
                     ['DEM_path'], paraDict['OUTPUT_PATH'], 6)
    # post.exportTxt(mask, mask, OUTPUT_PATH)


if __name__ == "__main__":
    # # LID index
    # # 1 Bio-cell retention;
    # # 2 Rain Garden;
    # # 3 Green Roof;
    # # 4 Infiltration Trent;
    # # 5 Pervious Pavement;
    # # 6 Rain Barrel;
    # # 7 vegetative swale;
    run(paraDict)
