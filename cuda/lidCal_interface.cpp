#include <torch/extension.h>
#include <vector>

// C++ interface at::Tensor Sat,   at::Tensor Sat,
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void lidCalculation_cuda(
    at::Tensor wetMask, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask, at::Tensor landuse_index,
    at::Tensor lidmask_index, at::Tensor areaMask, at::Tensor h, at::Tensor df, at::Tensor SurPara, at::Tensor SoilPara, at::Tensor StorPara,
    at::Tensor PavePara, at::Tensor DrainPara,
    at::Tensor DraMatPara, at::Tensor soilLimMin, at::Tensor soilLimMax,
    at::Tensor paveLimMax, at::Tensor storLimMax,
    at::Tensor cumuSurfaceWaterDepth,
    at::Tensor cumuSoilMoisture, at::Tensor cumuStorageWaterDepth,
    at::Tensor cumuPavementWaterDepth, at::Tensor drainrate, at::Tensor dx, 
    at::Tensor dy, at::Tensor dt);

void lidCalculation(at::Tensor wetMask, at::Tensor h_update, at::Tensor landuseMask, at::Tensor lidMask, at::Tensor landuse_index,
                    at::Tensor lidmask_index, at::Tensor areaMask, at::Tensor h, at::Tensor df, at::Tensor SurPara, at::Tensor SoilPara,
                    at::Tensor StorPara, at::Tensor PavePara, at::Tensor DrainPara,
                    at::Tensor DraMatPara, at::Tensor soilLimMin, at::Tensor soilLimMax,
                    at::Tensor paveLimMax, at::Tensor storLimMax, at::Tensor cumuSurfaceWaterDepth,
                    at::Tensor cumuSoilMoisture, at::Tensor cumuStorageWaterDepth,
                    at::Tensor cumuPavementWaterDepth, at::Tensor drainrate, at::Tensor dx, at::Tensor dy, at::Tensor dt)
{
  CHECK_INPUT(wetMask);
  CHECK_INPUT(landuseMask);
  CHECK_INPUT(lidMask);
  CHECK_INPUT(areaMask);
  CHECK_INPUT(landuse_index);
  CHECK_INPUT(lidmask_index);
  CHECK_INPUT(h);
  CHECK_INPUT(h_update);
  CHECK_INPUT(dt);
  CHECK_INPUT(SurPara);
  CHECK_INPUT(StorPara);
  CHECK_INPUT(PavePara);
  CHECK_INPUT(DrainPara);
  CHECK_INPUT(DraMatPara);
  CHECK_INPUT(soilLimMin);
  CHECK_INPUT(soilLimMax);
  CHECK_INPUT(paveLimMax);
  CHECK_INPUT(storLimMax);
  CHECK_INPUT(cumuSurfaceWaterDepth);
  CHECK_INPUT(cumuSoilMoisture);
  CHECK_INPUT(cumuStorageWaterDepth);
  CHECK_INPUT(cumuPavementWaterDepth);
  CHECK_INPUT(drainrate);

  lidCalculation_cuda(wetMask, h_update, landuseMask, lidMask, landuse_index, lidmask_index, 
                      areaMask, h, df, SurPara, SoilPara,
                      StorPara, PavePara, DrainPara, DraMatPara, soilLimMin, soilLimMax,
                      paveLimMax, storLimMax, cumuSurfaceWaterDepth, cumuSoilMoisture,
                      cumuStorageWaterDepth, cumuPavementWaterDepth, drainrate, dx, dy, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("addLidcalculation", &lidCalculation,
        "Lid_Infiltration_Calculation, CUDA version");
}

//at::Tensor drainrate, 