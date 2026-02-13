from posixpath import split
import torch
import glob
import os
import rasterio as rio
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import colors
import seaborn as sns
import numpy as np
from rasterio.windows import Window
from hipims_io import spatial_analysis as hpio_sa
try:
    from preProcessing import *
except ImportError:
    from .preProcessing import *

# def exportLidData(DEM_path, outPutPath):
#     from glob import glob
#     result_list = glob(outPutPath + '/*.pt')
#     result_list_sur = glob(outPutPath + '/Surlid*.pt')
#     result_list_soil = glob(outPutPath + '/SoilMoilid*.pt')
#     result_list_stor = glob(outPutPath + '/Storlid*.pt')
#     result_list_sur.sort()
#     result_list_soil.sort()
#     result_list_stor.sort()
#     device = torch.device("cuda",
#                           int(result_list[0][result_list[0].rfind(':') + 1]))

    # for i in range(len(result_list)):
    #     internal_data = torch.load(result_list[i])
    #     DATA = internal_data.to(torch.float32)
    #     data_cpu = DATA.cpu().numpy()

def exportRaster_gz(DEM_path, outPutPath, header_path, archive_pt=False):
    from glob import glob
    result_list = glob(outPutPath + '/*.pt')
    result_list_qx = glob(outPutPath + '/qx*.pt')
    result_list_qy = glob(outPutPath + '/qy*.pt')
    result_list_h = glob(outPutPath + '/h_tensor*.pt')

    result_list_h.sort()
    result_list_qx.sort()
    result_list_qy.sort()

    device = torch.device("cuda",
                          int(result_list[0][result_list[0].rfind(':') + 1]))
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)

    dem = dem.to(torch.float32)
    header= hpio_sa.arc_header_read(header_path)

    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()
    print(len(result_list_h))

    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i])
        dem[~mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'gz'
        hpio_sa.arcgridwrite(outPutName, data_cpu, header, compression=True)

        if (archive_pt == False):
            os.remove(result_list[i])

def exportRaster_gz_huv(DEM_path, outPutPath, header_path, archive_pt=False):
    from glob import glob
    result_list = glob(outPutPath + '/*.pt')
    result_list_u = glob(outPutPath + '/u*.pt')
    result_list_v = glob(outPutPath + '/v*.pt')
    result_list_h = glob(outPutPath + '/h_tensor*.pt')

    result_list_h.sort()
    result_list_u.sort()
    result_list_v.sort()

    device = torch.device("cuda",
                          int(result_list[0][result_list[0].rfind(':') + 1]))
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)

    dem = dem.to(torch.float32)
    header= hpio_sa.arc_header_read(header_path)

    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()
    print(len(result_list_h))

    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i])
        dem[~mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'gz'
        hpio_sa.arcgridwrite(outPutName, data_cpu, header, compression=True)

        if (archive_pt == False):
            os.remove(result_list[i])
# end

def exportRaster_tiff(DEM_path, outPutPath):
    from glob import glob
    result_list = glob(outPutPath + '/*.pt')
    result_list_qx = glob(outPutPath + '/qx*.pt')
    result_list_qy = glob(outPutPath + '/qy*.pt')
    result_list_h = glob(outPutPath + '/h_tensor*.pt')
    result_list_sur = glob(outPutPath + '/sur*.pt')
    result_list_soil = glob(outPutPath + '/soil*.pt')
    result_list_stor = glob(outPutPath + '/stor*.pt')
    

    result_list_h.sort()
    result_list_qx.sort()
    result_list_qy.sort()
    result_list_sur.sort()
    result_list_soil.sort()
    result_list_stor.sort()

    device = torch.device("cuda",
                          int(result_list[0][result_list[0].rfind(':') + 1]))
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)

    dem = dem.to(torch.float32)
    z = dem.clone()

    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()
    print(len(result_list_h))
    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i], device)
        dem[~mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        DATA_meta = demMeta.copy()
        DATA_meta.update({'nodata': nodatavalue})
        DATA_meta.update({'dtype': np.float32})
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'tif'
        # print(outPutName)
        with rio.open(outPutName, 'w', **DATA_meta) as outf:
            outf.write(data_cpu, 1)


def exportTiff(dem, mask, outPutPath):
    from glob import glob
    print(outPutPath + '/*.pt')
    result_list = glob(outPutPath + '/*.pt')
    mask = mask > 0
    mask_cpu = mask.cpu().numpy()
    dem = dem.to(torch.float32)
    for i in range(len(result_list)):
        internal_data = torch.load(result_list[i])
        dem[mask] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        DATA_meta = {}
        DATA_meta.update({'dtype': np.float32})
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'tif'
        # print(outPutName)
        # with rio.open(outPutName, 'w', **DATA_meta) as outf:
        #     outf.write(data_cpu, 1)
        with rio.open(outPutName,
                      'w',
                      driver='GTiff',
                      width=1620,
                      height=1000,
                      count=1,
                      dtype=np.float32,
                      nodata=nodatavalue) as dst:
            dst.write(data_cpu, indexes=1)


def ave_loadStep(z, size):
    step = np.zeros(size - 1, dtype=np.int32)
    z_cpu = z.cpu().numpy()
    count = np.count_nonzero(z_cpu)
    internal_index = np.argwhere(z_cpu > 0)
    for i in range(size - 1):
        step[i] = internal_index[
            int(float(i + 1) / float(size) * float(count)), 0] - 1
    return step


def multi_exportRaster_tiff(DEM_path, outPutPath, GPU_num):

    from glob import glob
    result_list = glob(outPutPath + '/*cuda:0*.pt')

    # for g_id in range(GPU_num):

    # device = torch.device("cuda",
    #                       int(result_list[0][result_list[0].rfind(':') + 1]))
    device = torch.device("cuda", 0)
    dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)
    # dem = dem.to(torch.double)

    z = dem.clone()

    step = ave_loadStep(mask, GPU_num)

    mask = ~mask
    print(demMeta)
    mask_cpu = mask.cpu().numpy()

    # dem = dem.to(torch.double)
    for i in range(len(result_list)):
        for j in range(GPU_num):
            file_path = result_list[i][:result_list[i].rfind(':') + 1] + str(
                j) + result_list[i][result_list[i].rfind(':') + 2:]
            internal_data = torch.load(file_path).to(device)
            if j == 0:
                temp_mask = (~mask).clone()
                temp_dem = dem.clone()
                temp_mask[step[0] + 2:, :] = False
                temp_dem[temp_mask] = internal_data.to(torch.float32)
                dem[:step[0], :] = temp_dem[:step[0], :]
                del temp_mask, temp_dem
            elif j == GPU_num - 1:
                temp_mask = (~mask).clone()
                temp_dem = dem.clone()
                temp_mask[:step[-1] - 2, :] = False
                temp_dem[temp_mask] = internal_data.to(torch.float32)
                dem[step[-1]:, :] = temp_dem[step[-1]:, :]
                del temp_mask, temp_dem
            else:
                temp_mask = (~mask).clone()
                temp_dem = dem.clone()
                temp_mask[:step[j - 1] - 2, :] = False
                temp_mask[step[j] + 2:, :] = False
                temp_dem[temp_mask] = internal_data.to(torch.float32)
                dem[step[j - 1]:step[j], :] = temp_dem[step[j - 1]:step[j], :]
                del temp_mask, temp_dem

        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        DATA_meta = demMeta.copy()
        DATA_meta.update({'nodata': nodatavalue})
        DATA_meta.update({'dtype': np.float32})
        topAddress = result_list[i][:result_list[i].rfind('_')]
        timeAddress = result_list[i][result_list[i].find('[') +
                                     1:result_list[i].find(']')]
        outPutName = topAddress + "_" + timeAddress + 'tif'
        print(outPutName)
        with rio.open(outPutName, 'w', **DATA_meta) as outf:
            outf.write(data_cpu, 1)
    

# def cutSectPlot(DEM_path, outPutPath, secIndex):
#     from glob import glob
#     import re
#     result_list_qx = glob(outPutPath + '/qx*.pt')
#     result_list_h = glob(outPutPath + '/h_tensor*.pt')
#     # int(result_list_h[0].split('[')[1][0:].split('.')[0][0:])
#     result_list_h = sorted(result_list_h, key = lambda i:(int(i.split('[')[1][0:].split('.')[0][0:])))
    
#     result_list_qx = sorted(result_list_qx, key = lambda i:(int(i.split('[')[1][0:].split('.')[0][0:])))

#     device = torch.device("cuda", 1)
#     dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)
#     mask = ~mask
#     print(demMeta)
#     mask_cpu = mask.cpu().numpy()
#     sec_dataqx=[[] for i in range(len(result_list_qx))]
#     sec_datah=[[] for i in range(len(result_list_h))]
#     for i in range(len(result_list_qx)):
#         internal_data = torch.load(result_list_qx[i])
#         dem[~mask] = internal_data.to(torch.float32)
#         # print(dem)
#         data_cpu = dem.cpu().numpy()
#         data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
#         nodatavalue = -9999.
#         data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
#         sec_dataqx[i] = data_cpu[secIndex,:]
    
#     for i in range(len(result_list_h)):
#         internal_data = torch.load(result_list_h[i])
#         dem[~mask] = internal_data.to(torch.float32)
#         # print(dem)
#         data_cpu = dem.cpu().numpy()
#         data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
#         nodatavalue = -9999.
#         data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
#         sec_datah[i] = data_cpu[secIndex,:]

#     sec_dataqx=np.array(sec_dataqx)
#     sec_datah=np.array(sec_datah)
#     # sec_datau=np.array(sec_datau)
#     np.savetxt(outPutPath + "/qx_sec" + '.txt', sec_dataqx)
#     np.savetxt(outPutPath + "/h_sec" + '.txt', sec_datah)
    # np.savetxt(outPutPath + "/u_sec" + '.txt', sec_datau)


def cutSectPlot(dem, mask, outPutPath, secIndex):
    from glob import glob
    import re
    result_list_qx = glob(outPutPath + '/qx*.pt')
    result_list_h = glob(outPutPath + '/h_tensor*.pt')
    # int(result_list_h[0].split('[')[1][0:].split('.')[0][0:])
    result_list_h = sorted(result_list_h, key = lambda i:(int(i.split('[')[1][0:].split('.')[0][0:])))
    
    result_list_qx = sorted(result_list_qx, key = lambda i:(int(i.split('[')[1][0:].split('.')[0][0:])))

    device = torch.device("cuda", 3)
    # dem, mask, mask_boundary, demMeta = importDEMData(DEM_path, device)
    # mask = ~mask
    # print(demMeta)
    mask_cpu = mask.cpu().numpy()
    sec_dataqx=[[] for i in range(len(result_list_qx))]
    sec_datah=[[] for i in range(len(result_list_h))]
    for i in range(len(result_list_qx)):
        internal_data = torch.load(result_list_qx[i])
        dem[mask>0] = internal_data.to(torch.double)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        sec_dataqx[i] = data_cpu[secIndex,:]
    
    for i in range(len(result_list_h)):
        internal_data = torch.load(result_list_h[i])
        dem[mask>0] = internal_data.to(torch.float32)
        # print(dem)
        data_cpu = dem.cpu().numpy()
        data_cpu = ma.masked_array(data_cpu, mask=mask_cpu)
        nodatavalue = -9999.
        data_cpu = ma.filled(data_cpu, fill_value=nodatavalue)
        sec_datah[i] = data_cpu[secIndex,:]

    sec_dataqx=np.array(sec_dataqx)
    sec_datah=np.array(sec_datah)
    # sec_datau=np.array(sec_datau)
    np.savetxt(outPutPath + "/qx_sec" + '.txt', sec_dataqx)
    np.savetxt(outPutPath + "/h_sec" + '.txt', sec_datah)
    # np.savetxt(outPutPath + "/u_sec" + '.txt', sec_datau)

case = 2

if __name__ == "__main__":
    if case == 0:
        exportRaster_tiff('/home/cvjz3/CanTho/raster/DEM.tif',
                          '/home/cvjz3/CanTho/output')
    elif case == 1:
        from glob import glob
        result_list = glob(
            '/home/cvjz3/steadyFlow/jh_1st/5/output/h_tensor*.pt')
        # result_list += glob('/home/cvjz3/steadyFlow/output' +
        #                         '/wl_tensor*.pt')
        result_list += glob(
            '/home/cvjz3/steadyFlow/jh_1st/5/output/qx_tensor*.pt')
        device = torch.device("cuda", 1)

        x = range(240)
        for i in range(len(result_list)):
            internal_data = torch.load(result_list[i])
            internal_data = internal_data.resize(8, 240)  # print(dem)
            sec_data = internal_data[0]
            sec_data = sec_data.cpu().numpy()
            fig, ax = plt.subplots()
            plt.plot(x, sec_data)
            plt.ylim((0., 0.02))
            plt.savefig(result_list[i][:-2] + 'png')
    else:
        multi_exportRaster_tiff(
            '/home/cv/cvjz3/Eden/Tiff_Data/Tiff/DEM.tif',
            '/home/cv/cvjz3/Eden/output_whole', 2)
        # multi_exportRaster_tiff(
        #     '/home/cvjz3/Eden/Tiff_Data/Tiff/DEM.tif',
        #     '/home/cvjz3/Eden/output', 2)
