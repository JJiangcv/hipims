from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 1. 定义编译参数
# 修改：移除硬编码的架构参数，改用环境变量 TORCH_CUDA_ARCH_LIST 控制
# 修改：将 std=c++11 改为 std=c++14 以更好匹配 PyTorch 1.x
EXTRA_COMPILE_ARGS = {
    'cxx': ['-std=c++14', '-O2', '-Wall', '-g'],
    'nvcc': [
        '-std=c++14',
        '--expt-extended-lambda',
        '--use_fast_math',
        '-Xcompiler', '-Wall', '-g'
    ],
}

# 2. 定义扩展模块
# (列表保持不变，直接复制您的文件名)
ext_modules = [
    CUDAExtension('euler_update', [
        'euler_update.cpp',
        'euler_update_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('fluxCalculation_jh_modified_surface', [
        'fluxCal_jh_modified_surface.cpp',
        'fluxCal_jh_modified_surface_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('fluxCal_2ndOrder_jh_improved', [
        'fluxCal_2ndOrder_jh_improved.cpp',
        'flucCal_2ndOrder_jh_improved_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('frictionCalculation', [
        'friction_interface.cpp',
        'frictionCUDA_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('frictionCalculation_implicit', [
        'friction_implicit_interface.cpp',
        'friction_implicit_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('friction_implicit_andUpdate_jh', [
        'friction_implicit_andUpdate_jh_interface.cpp',
        'friction_implicit_andUpdate_jh_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('infiltration_sewer', [
        'infiltration_interface.cpp',
        'infiltrationCUDA_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('station_PrecipitationCalculation', [
        'stationPrecipitation_interface.cpp',
        'stationPrecipitation_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('timeControl', [
        'timeControl.cpp',
        'timeControl_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('fluxMask', [
        'fluxMaskGenerator.cpp',
        'fluxMaskGenerator_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('fluxMask_jjh_modified', [
        'fluxMaskGenerator_jjh_modified.cpp',
        'fluxMaskGenerator_Kernel_jjh_modified.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('lidInfil', [
        'lidInfiltration_interface.cpp',
        'lidInfiltration_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('lidCal', [
        'lidCal_interface.cpp',
        'lidCalCUDA_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),

    CUDAExtension('lidInfil_new', [
        'lidinfil_new_interface.cpp',
        'lidinfil_new_Kernel.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('lidCal_new', [
        'lidCalCUDANew_Kernel_interface.cpp',
        'lidCalCUDANew_Kernelnew.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
    
    CUDAExtension('lidFlux', [
        'fluxCal_lid_interface.cpp',
        'fluxCal_lid.cu',
    ], extra_compile_args=EXTRA_COMPILE_ARGS),
]

setup(
    name='hipims',
    version='1.1.0',
    description='PyTorch implementation of "HiPIMS"',
    author='Jiaheng Zhao',
    author_email='j.zhao@lboro.ac.uk',
    license='in CopyRight: in-house code',
    packages=find_packages(),  # 关键修正：自动寻找包
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)