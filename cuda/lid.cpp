
#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include "infil.h"
#include "lid.h"
using namespace std;

#define BIG 1.E10                            // Generic large value
#define MIN(x, y) (((x) <= (y)) ? (x) : (y)) /* minimum of x and y    */
#define MAX(x, y) (((x) >= (y)) ? (x) : (y)) /* maximum of x and y    */
#define ZERO 1.E-10                          // Effective zero value
#define STOPTOL 0.001                        // integration error tolerance in ft (= 1 mm)
#define TRUE 1                               // Value for TRUE state
#define FALSE 0                              // Value for FALSE state

static TLidProc theLidProc;   // ptr. to a LID process
static double NativeInfil;    // native soil infil. rate (m/s)
static double MaxNativeInfil; // native soil infil. rate limit (m/s)
static double Tstep;          // current time step (sec)
static double SurfaceInflow;  // precip. + runon to LID unit (ft/s)
static double SurfaceInfil;   // infil. rate from surface layer (ft/s)
static double SurfaceOutflow; // outflow from surface layer (ft/s)
static double SurfaceVolume;  // volume in surface storage (ft)

static double PavePerc;   // percolation from pavement layer (ft/s)
static double PaveVolume; // volume stored in pavement layer  (ft)

static double SoilPerc;   // percolation from soil layer (ft/s)
static double SoilVolume; // volume in soil/pavement storage (ft)

static double StorageInflow; // inflow rate to storage layer (ft/s)
static double StorageExfil;  // exfil. rate from storage layer (ft/s)
static double StorageDrain;  // underdrain flow rate layer (ft/s)
static double StorageVolume; // volume in storage layer (ft)

static double Xold[MAX_LAYERS]; // previous moisture level in LID layers

static double exflowintotal;

double RFI;

static int M;
static int N;
double dx, dy;

char *LidLayerWords[] =
    {"SURFACE", "SOIL", "STORAGE", "PAVEMENT", "DRAINMAT", "DRAIN",
     "REMOVALS", NULL};

double getSoilPercRate(double theta)
//
//  Purpose: computes percolation rate of water through a LID's soil layer.
//  Input:   theta = moisture content (fraction)
//  Output:  returns percolation rate within soil layer (ft/s)
//
{
    double delta; // moisture deficit

    // ... no percolation if soil moisture <= field capacity
    if (theta <= theLidProc.lidPram.soil.fieldCap)
        return 0.0;

    // ... perc rate = unsaturated hydraulic conductivity
    delta = theLidProc.lidPram.soil.porosity - theta;
    return theLidProc.lidPram.soil.kSat * exp(-delta * theLidProc.lidPram.soil.kSlope);
}

double getStorageExfilRate()
//
//  Purpose: computes exfiltration rate from storage zone into
//           native soil beneath a LID.
//  Input:   depth = depth of water storage zone (ft)
//  Output:  returns infiltration rate (ft/s)
//
{
    double infil = 0.0;
    double clogFactor = 0.0;

    if (theLidProc.lidPram.storage.kSat == 0.0)
        return 0.0;
    if (MaxNativeInfil == 0.0)
        return 0.0;

    //... reduction due to clogging
    clogFactor = theLidProc.lidPram.storage.clogFactor;
    if (clogFactor > 0.0)
    {
        //----------------------------------------------------------------------------------
        //clogFactor = theLidUnit.waterBalance.inflow / clogFactor;
        clogFactor = MIN(clogFactor, 1.0);
    }

    //... infiltration rate = storage Ksat reduced by any clogging
    infil = theLidProc.lidPram.storage.kSat * (1.0 - clogFactor);

    //... limit infiltration rate by any groundwater-imposed limit
    return MIN(infil, MaxNativeInfil);
}

double getStorageDrainRate(double storageDepth, double soilTheta,
                           double paveDepth, double surfaceDepth)
//
//  Purpose: computes underdrain flow rate in a LID's storage layer.
//  Input:   storageDepth = depth of water in storage layer (ft)
//           soilTheta    = moisture content of soil layer
//           paveDepth    = effective depth of water in pavement layer (ft)
//           surfaceDepth = depth of ponded water on surface layer (ft)
//  Output:  returns flow in underdrain (ft/s)
//
//  Note:    drain eqn. is evaluated in user's units.
//  Note:    head on drain is water depth in storage layer plus the
//           layers above it (soil, pavement, and surface in that order)
//           minus the drain outlet offset.
{
    int curve = theLidProc.lidPram.drain.qCurve; //(5.1.013)
    double head = storageDepth;
    double outflow = 0.0;
    double paveThickness = theLidProc.lidPram.pavement.thickness;
    double soilThickness = theLidProc.lidPram.soil.thickness;
    double soilPorosity = theLidProc.lidPram.soil.porosity;
    double soilFieldCap = theLidProc.lidPram.soil.fieldCap;
    double storageThickness = theLidProc.lidPram.storage.thickness;

    // --- storage layer is full
    if (storageDepth >= storageThickness)
    {
        // --- a soil layer exists
        if (soilThickness > 0.0)
        {
            // --- increase head by fraction of soil layer saturated
            if (soilTheta > soilFieldCap)
            {
                head += (soilTheta - soilFieldCap) /
                        (soilPorosity - soilFieldCap) * soilThickness;

                // --- soil layer is saturated, increase head by water
                //     depth in layer above it
                if (soilTheta >= soilPorosity)
                {
                    if (paveThickness > 0.0)
                        head += paveDepth;
                    else
                        head += surfaceDepth;
                }
            }
        }

        // --- no soil layer so increase head by water level in pavement
        //     layer and possibly surface layer
        if (paveThickness > 0.0)
        {
            head += paveDepth;
            if (paveDepth >= paveThickness)
                head += surfaceDepth;
        }
    }

    // --- no outflow if:                                                      //(5.1.013)
    //     a) no prior outflow and head below open threshold                   //
    //     b) prior outflow and head below closed threshold                    //
    //---------------------------------------------------------------------------------------------
    if (theLidProc.oldDrainFlow == 0.0 && //
        head <= theLidProc.lidPram.drain.hOpen)
        return 0.0;                      //
    if (theLidProc.oldDrainFlow > 0.0 && //
        head <= theLidProc.lidPram.drain.hClose)
        return 0.0; //

    // --- make head relative to drain offset
    head -= theLidProc.lidPram.drain.offset;

    // --- compute drain outflow from underdrain flow equation in user units
    //     (head in inches or mm, flow rate in in/hr or mm/hr)
    if (head > ZERO)
    {
        // --- convert head to user units
        //head *= UCF(RAINDEPTH);

        // --- compute drain outflow in user units
        outflow = theLidProc.lidPram.drain.coeff *
                  pow(head, theLidProc.lidPram.drain.expon);

        // --- apply user-supplied control curve to outflow
        //if (curve >= 0)  outflow *= table_lookup(&Curve[curve], head);         //(5.1.013)

        // --- convert outflow to ft/s
        outflow /= RFI;
    }
    return outflow;
}

int modpuls_solve(int n, double *x, double *xOld, double *xPrev,
                  double *xMin, double *xMax, double *xTol,
                  double *qOld, double *q, double dt, double omega,
                  void (*derivs)(double *, double *))
//
//  Purpose: solves system of equations dx/dt = q(x) for x at end of time step
//           dt using a modified Puls method.
//  Input:   n = number of state variables
//           x = vector of state variables
//           xOld = state variable values at start of time step
//           xPrev = state variable values from previous iteration
//           xMin = lower limits on state variables
//           xMax = upper limits on state variables
//           xTol = convergence tolerances on state variables
//           qOld = flux rates at start of time step
//           q = flux rates at end of time step
//           dt = time step (sec)
//           omega = time weighting parameter (use 0 for Euler method
//                   or 0.5 for modified Puls method)
//           derivs = pointer to function that computes flux rates q as a
//                    function of state variables x
//  Output:  returns number of steps required for convergence (or 0 if
//           process doesn't converge)
//
{
    int i;
    int canStop;
    int steps = 1;
    int maxSteps = 20;

    //... initialize state variable values
    for (i = 0; i < n; i++)
    {
        xOld[i] = x[i];
        xPrev[i] = x[i];
    }

    //... repeat until convergence achieved
    while (steps < maxSteps)
    {
        //... compute flux rates for current state levels
        canStop = 1;
        derivs(x, q);

        //... update state levels based on current flux rates
        for (i = 0; i < n; i++)
        {
            x[i] = xOld[i] + (omega * qOld[i] + (1.0 - omega) * q[i]) * dt;
            x[i] = MIN(x[i], xMax[i]);
            x[i] = MAX(x[i], xMin[i]);

            if (omega > 0.0 &&
                fabs(x[i] - xPrev[i]) > xTol[i])
                canStop = 0;
            xPrev[i] = x[i];
        }

        //... return if process converges
        if (canStop)
            return steps;
        steps++;
    }

    //... no convergence so return 0
    return 0;
}

void biocellFluxRates(double x[], double f[])
//
//  Purpose: computes flux rates from the layers of a bio-retention cell LID.
//  Input:   x = vector of storage levels
//  Output:  f = vector of flux rates
//
{
    // Moisture level variables
    double surfaceDepth;
    double soilTheta;
    double storageDepth;

    // Intermediate variables
    double availVolume;
    double maxRate;

    // LID layer properties
    double soilThickness = theLidProc.lidPram.soil.thickness;
    double soilPorosity = theLidProc.lidPram.soil.porosity;
    double soilFieldCap = theLidProc.lidPram.soil.fieldCap;
    double soilWiltPoint = theLidProc.lidPram.soil.wiltPoint;
    double storageThickness = theLidProc.lidPram.storage.thickness;
    double storageVoidFrac = theLidProc.lidPram.storage.voidFrac;

    //... retrieve moisture levels from input vector
    surfaceDepth = x[SURF];
    soilTheta = x[SOIL];
    storageDepth = x[STOR];

    //... convert moisture levels to volumes
    SurfaceVolume = surfaceDepth * theLidProc.lidPram.surface.voidFrac;
    SoilVolume = soilTheta * soilThickness;
    StorageVolume = storageDepth * storageVoidFrac;

    //-----------------------------------------------------------------------------------------
    //... get ET rates
    /*availVolume = SoilVolume - soilWiltPoint * soilThickness;
	getEvapRates(SurfaceVolume, 0.0, availVolume, StorageVolume, 1.0);
	if (soilTheta >= soilPorosity) StorageEvap = 0.0;*/

    //... soil layer perc rate
    SoilPerc = getSoilPercRate(soilTheta);

    //... limit perc rate by available water
    availVolume = (soilTheta - soilFieldCap) * soilThickness;
    //--------------------------------------------------
    maxRate = MAX(availVolume, 0.0) / Tstep; // -SoilEvap;
    SoilPerc = MIN(SoilPerc, maxRate);
    SoilPerc = MAX(SoilPerc, 0.0);

    //... exfiltration rate out of storage layer
    //------------------------------------------------------------------------------------------
    StorageExfil = getStorageExfilRate();

    //... underdrain flow rate
    StorageDrain = 0.0;
    if (theLidProc.lidPram.drain.coeff > 0.0)
    {
        StorageDrain = getStorageDrainRate(storageDepth, soilTheta, 0.0,
                                           surfaceDepth);
    }

    //... special case of no storage layer present
    if (storageThickness == 0.0)
    {
        //StorageEvap = 0.0;
        maxRate = MIN(SoilPerc, StorageExfil);
        SoilPerc = maxRate;
        StorageExfil = maxRate;

        //... limit surface infil. by unused soil volume
        //-------------------------------------------------------------------------
        maxRate = (soilPorosity - soilTheta) * soilThickness / Tstep +
                  SoilPerc; //;
        SurfaceInfil = MIN(SurfaceInfil, maxRate);
    }

    //... storage & soil layers are full
    else if (soilTheta >= soilPorosity && storageDepth >= storageThickness)
    {
        //... limiting rate is smaller of soil perc and storage outflow
        maxRate = StorageExfil + StorageDrain;
        if (SoilPerc < maxRate)
        {
            maxRate = SoilPerc;
            if (maxRate > StorageExfil)
                StorageDrain = maxRate - StorageExfil;
            else
            {
                StorageExfil = maxRate;
                StorageDrain = 0.0;
            }
        }
        else
            SoilPerc = maxRate;

        //... apply limiting rate to surface infil.
        SurfaceInfil = MIN(SurfaceInfil, maxRate);
    }

    //... either layer not full
    else if (storageThickness > 0.0)
    {
        //... limit storage exfiltration by available storage volume
        //--------------------------------------------------------------------------
        maxRate = SoilPerc + storageDepth * storageVoidFrac / Tstep; // -storageEvap
        StorageExfil = MIN(StorageExfil, maxRate);
        StorageExfil = MAX(StorageExfil, 0.0);

        //--------------------------------------------------------------------------
        //... limit underdrain flow by volume above drain offset
        if (StorageDrain > 0.0)
        {
            maxRate = -StorageExfil; //- StorageEvap
            if (storageDepth >= storageThickness)
                maxRate += SoilPerc;
            if (theLidProc.lidPram.drain.offset <= storageDepth)
            {
                maxRate += (storageDepth - theLidProc.lidPram.drain.offset) *
                           storageVoidFrac / Tstep;
            }
            maxRate = MAX(maxRate, 0.0);
            StorageDrain = MIN(StorageDrain, maxRate);
        }

        //... limit soil perc by unused storage volume
        //-----------------------------------------------------------------------------------
        maxRate = StorageExfil + StorageDrain +
                  (storageThickness - storageDepth) *
                      storageVoidFrac / Tstep; // + StorageEvap
        SoilPerc = MIN(SoilPerc, maxRate);

        //... limit surface infil. by unused soil volume
        //---------------------------------------------------------------------------------------
        maxRate = (soilPorosity - soilTheta) * soilThickness / Tstep +
                  SoilPerc; // +SoilEvap;
        SurfaceInfil = MIN(SurfaceInfil, maxRate);
    }

    //... find surface layer outflow rate
    //-----in our model the outflow rate is given by the hipims has included in the inflow
    //SurfaceOutflow = getSurfaceOutflowRate(surfaceDepth,i);

    //... compute overall layer flux rates
    f[SURF] = (SurfaceInflow - SurfaceInfil) /
              theLidProc.lidPram.surface.voidFrac; //- SurfaceEvap- SurfaceOutflow
    f[SOIL] = (SurfaceInfil - SoilPerc) /
              theLidProc.lidPram.soil.thickness; //- SoilEvap
    if (storageThickness == 0.0)
        f[STOR] = 0.0;
    else
        f[STOR] = (SoilPerc - StorageExfil - StorageDrain) /
                  theLidProc.lidPram.storage.voidFrac; //- StorageEvap
}

//======================================================================================//
//  Purpose: initialize th lid parameters
void ini_lidProcPram(TLidProc *LidProc, int i)
{
    LidProc[i].lidPram.surface.thickness = 0;
    LidProc[i].lidPram.soil.thickness = 0;
    LidProc[i].lidPram.drain.coeff = 0;
    LidProc[i].lidPram.drainMat.thickness = 0;
    LidProc[i].lidPram.pavement.thickness = 0;
    LidProc[i].lidPram.storage.thickness = 0;
}

//  Purpose: finds match between string and array of keyword strings.
int findmatch(string s)
//  Input:   s = character string
//  Output:  returns index of matching keyword or -1 if no match found
{
    int i = 0;
    while (s != "NULL")
    {
        if (s == LidLayerWords[i])
            return (i);
        i++;
    }
    return (-1);
}

//==========================================================================================================================//
void readlidPara(TLidProc *LidProc, int m, int n)
// Input:
//		M: row number; N: column number
// Output: none
// Purpose: reads input file to determin input parameters for each LID controls
{
    M = m;
    N = n;

    int ind;
    stringstream ss; // read file name
    string line;     // input line content
    ifstream myfile; // operation on file
    myfile.precision(15);
    ss << "/home/jinghuajiang/LidTest_case/input/lidLocation.dat";
    string strFile = ss.str();
    myfile.open(strFile.c_str());
    if (myfile.is_open())
    {
        getline(myfile, line);
        getline(myfile, line);
        getline(myfile, line);
        for (int i = 0; i < M * N; i++)
        {
            myfile >> ind;
            myfile >> LidProc[i].lidType;
            myfile >> LidProc[i].lidID;
        }
    }
    else
    {
        printf("Cannot Open lidLocation File");
    }

    ss.str("");
    ss.clear();
    ss.clear();
    for (int i = 0; i < M * N; i++)
    {
        if (LidProc[i].lidID >= 0)
        {
            ini_lidProcPram(LidProc, i);
            int m; // finds match between string and array of keyword strings
            int layerNum;
            int layerid = LidProc[i].lidID;
            ss << "/home/jinghuajiang/LidTest_case/input/";
            ss << "lidPram_";
            ss << layerid;
            ss << ".dat";
            string strFile1 = ss.str();
            ifstream myfile1;
            myfile1.precision(15);
            myfile1.open(strFile1.c_str());
            if (myfile1.is_open())
            {
                getline(myfile1, line);
                myfile1 >> layerNum;
                getline(myfile1, line);
                getline(myfile1, line);
                for (int j = 1; j <= layerNum; j++)
                {
                    string layer;
                    myfile1 >> layer;
                    m = findmatch(layer);
                    switch (m)
                    {
                    case SURF:
                        myfile1 >> LidProc[i].lidPram.surface.thickness;
                        myfile1 >> LidProc[i].lidPram.surface.voidFrac;
                        myfile1 >> LidProc[i].lidPram.surface.roughness;
                        myfile1 >> LidProc[i].lidPram.surface.surfSlope;
                        myfile1 >> LidProc[i].lidPram.surface.sideSlope;
                        break;
                    case SOIL:
                        myfile1 >> LidProc[i].lidPram.soil.thickness;
                        myfile1 >> LidProc[i].lidPram.soil.porosity;
                        myfile1 >> LidProc[i].lidPram.soil.fieldCap;
                        myfile1 >> LidProc[i].lidPram.soil.wiltPoint;
                        myfile1 >> LidProc[i].lidPram.soil.kSat;
                        myfile1 >> LidProc[i].lidPram.soil.kSlope;
                        myfile1 >> LidProc[i].lidPram.soil.suction;
                        LidProc[i].initSat = 0.0;
                        LidProc[i].soilInfil.Ks = LidProc[i].lidPram.soil.kSat;
                        LidProc[i].soilInfil.S = LidProc[i].lidPram.soil.suction;
                        LidProc[i].soilInfil.IMDmax = (LidProc[i].lidPram.soil.porosity -
                                                       LidProc[i].lidPram.soil.wiltPoint) *
                                                      (1.0 - LidProc[i].initSat);
                        LidProc[i].soilInfil.Lu = (1 / 3.0) * sqrt(LidProc[i].soilInfil.Ks * 3600 * 1000 * 43200.0 / 1097280.0) * 0.3048;
                        break;
                    case STOR:
                        myfile1 >> LidProc[i].lidPram.storage.thickness;
                        myfile1 >> LidProc[i].lidPram.storage.voidFrac;
                        myfile1 >> LidProc[i].lidPram.storage.kSat;
                        myfile1 >> LidProc[i].lidPram.storage.clogFactor;
                        break;
                    case PAVE:
                        myfile1 >> LidProc[i].lidPram.pavement.thickness;
                        myfile1 >> LidProc[i].lidPram.pavement.voidFrac;
                        myfile1 >> LidProc[i].lidPram.pavement.impervFrac;
                        myfile1 >> LidProc[i].lidPram.pavement.kSat;
                        myfile1 >> LidProc[i].lidPram.pavement.clogFactor;
                        myfile1 >> LidProc[i].lidPram.pavement.regenDays;
                        myfile1 >> LidProc[i].lidPram.pavement.regenDegree;
                        break;
                    case DRAIN:
                        myfile1 >> LidProc[i].lidPram.drain.coeff;
                        myfile1 >> LidProc[i].lidPram.drain.expon;
                        myfile1 >> LidProc[i].lidPram.drain.offset;
                        myfile1 >> LidProc[i].lidPram.drain.delay;
                        myfile1 >> LidProc[i].lidPram.drain.hOpen;
                        myfile1 >> LidProc[i].lidPram.drain.hClose;
                        break;
                    }
                }
            }
            else
            {
                printf("Cannot Open LidParameter File");
            }
        }
    }
}

void lid_initialize(TLidProc *LidProc)
{
    // input: Row Number: M; Column Number: N;
    // output: none
    // purpose: initialize the lid parameters
    for (int i = 0; i < M * N; i++)
    {
        LidProc[i].surfaceDepth = 0.0;
        LidProc[i].paveDepth = 0.0;
        LidProc[i].soilMoisture = 0.0;
        LidProc[i].storageDepth = 0.0;
        LidProc[i].exflowintotal = 0;
        LidProc[i].NativeInfil = 0.0;
        LidProc[i].MaxNativeInfil = 0.0;
        LidProc[i].h_inLid = 0.0;

        if (LidProc[i].lidPram.soil.thickness > 0.0)
        {
            LidProc[i].soilMoisture = LidProc[i].lidPram.soil.wiltPoint +
                                      LidProc[i].initSat * (LidProc[i].lidPram.soil.porosity -
                                                            LidProc[i].lidPram.soil.wiltPoint);
        }
        if (LidProc[i].lidPram.storage.thickness > 0.0)
        {
            LidProc[i].storageDepth = LidProc[i].initSat * LidProc[i].lidPram.storage.thickness;
        }

        LidProc[i].oldFluxRates[0] = 0.0;
        LidProc[i].oldFluxRates[1] = 0.0;
        LidProc[i].oldFluxRates[2] = 0.0;
        LidProc[i].oldFluxRates[3] = 0.0;
        LidProc[i].oldFluxRates[4] = 0.0;

        if (LidProc[i].soilInfil.Ks > 0.0)
        {
            // initialise green ampt parameters
            grnampt_initState(&LidProc[i].soilInfil);
        }
    }
}

// Purpose: computes inflow treated by LID area (m/s)
double lid_getInflow(double *ptrqx_input, double *ptrqy_input, double dt, int i, double rainfall)

{
    RFI = rainfall;
    double lidinflow;
    if (ptrqx_input[i]>0)
    {
        if(ptrqy_input[i]>0)
        {
            lidinflow = sqrt(pow(ptrqx_input[i] * dy, 2) + pow(ptrqy_input[i] * dx, 2)) / (dx * dy);
        }
        else
        {
            lidinflow = sqrt(pow(ptrqx_input[i] * dy, 2) + 0.0) / (dx * dy);
        }
    }
    else
    {
        if (ptrqy_input[i]>0)
        {
            lidinflow = sqrt(0.0 + pow(ptrqy_input[i] * dx, 2)) / (dx * dy);
        }
        else
        {
            lidinflow = 0.0;
        }
        
    }
    //printf("qxInflow:%f\tqyInflow:%f\t",ptrqx_input[i],ptrqy_input[i]);
    lidinflow = lidinflow + RFI;
    //printf("getInflow:%f\t",lidinflow);
    return lidinflow;
}

// Purpose: computes the new surfacedepth (m)
void lidproc_getSurfaceDepth(TLidProc *LidProc, double lidInflow, double tStep, int index)
{
    int i;
    double x[MAX_LAYERS];     // layer moisture levels
    double xOld[MAX_LAYERS];  // work vector
    double xPrev[MAX_LAYERS]; // work vector
    double xMin[MAX_LAYERS];  // lower limit on moisture levels
    double xMax[MAX_LAYERS];  // upper limit on moisture levels
    double fOld[MAX_LAYERS];  // previously computed flux rates
    double f[MAX_LAYERS];     // newly computed flux rates

    double omega = 0.0; // integration time weighting

    // convergence tolerance on moisture levels (m, moisture fraction , m)
    double xTol[MAX_LAYERS] = {STOPTOL, STOPTOL, STOPTOL, STOPTOL};

    //... define a pointer to function that computes flux rates through the LID
    void (*fluxRates)(double *, double *) = NULL;

    //... save references to the LID process and LID unit
    theLidProc = LidProc[index];

    //... save max. infil. & time step to shared variables
    MaxNativeInfil = theLidProc.MaxNativeInfil;
    Tstep = tStep;

    //... store current moisture levels in vector x
    x[SURF] = theLidProc.surfaceDepth;
    x[SOIL] = theLidProc.soilMoisture;
    x[STOR] = theLidProc.storageDepth;
    x[PAVE] = theLidProc.paveDepth;

    //... initialize layer moisture volumes, flux rates and moisture limits
    SurfaceVolume = 0.0;
    PaveVolume = 0.0;
    SoilVolume = 0.0;
    StorageVolume = 0.0;
    SurfaceInflow = lidInflow;
    SurfaceInfil = 0.0;
    SurfaceOutflow = 0.0;
    PavePerc = 0.0;
    SoilPerc = 0.0;
    StorageInflow = 0.0;
    StorageExfil = 0.0;
    StorageDrain = 0.0;

    for (int j = 0; j < MAX_LAYERS; j++)
    {
        f[j] = 0.0;
        fOld[j] = theLidProc.oldFluxRates[j];
        xMin[j] = 0.0;
        xMax[j] = BIG;
        Xold[j] = x[j];
    }

    //... find Green-Ampt infiltration from surface layer
    if (theLidProc.lidType == 5)
        SurfaceInfil = 0.0;
    else if (theLidProc.soilInfil.Ks > 0.0)
    {
        SurfaceInfil =
            grnampt_getInfil(&theLidProc.soilInfil, Tstep,
                             SurfaceInflow, theLidProc.surfaceDepth,
                             MOD_GREEN_AMPT);
        //printf("SurfaceInfil:%f\tTstep:%f\t", SurfaceInfil, Tstep);
    }
    else
        SurfaceInfil = theLidProc.NativeInfil;

    //... set moisture limits for soil & storage layers
    if (theLidProc.lidPram.soil.thickness > 0.0)
    {
        xMin[SOIL] = theLidProc.lidPram.soil.wiltPoint;
        xMax[SOIL] = theLidProc.lidPram.soil.porosity;
    }
    if (theLidProc.lidPram.pavement.thickness > 0.0)
    {
        xMax[PAVE] = theLidProc.lidPram.pavement.thickness;
    }
    if (theLidProc.lidPram.storage.thickness > 0.0)
    {
        xMax[STOR] = theLidProc.lidPram.storage.thickness;
    }
    if (theLidProc.lidType == 3)
    {
        xMax[STOR] = theLidProc.lidPram.drainMat.thickness;
    }

    //... determine which flux rate function to use
    switch (theLidProc.lidType)
    {
    case 1:
        fluxRates = &biocellFluxRates;
        break;
    case 2:
        fluxRates = &biocellFluxRates;
        break;
        //case 3:		fluxRates = &greenRoofFluxRates; break;
        //case 4:		fluxRates = &trenchFluxRates;    break;
        //case 5:		fluxRates = &pavementFluxRates;  break;
        //case 6:     fluxRates = &barrelFluxRates;    break;
        //case 7:     fluxRates = &swaleFluxRates;      break;
        //case 8:     fluxRates = &roofFluxRates;
        //omega = 0.5;
        //break;
        //default:              return 0.0;
    }

    //... update moisture levels and flux rates over the time step
    i = modpuls_solve(MAX_LAYERS, x, xOld, xPrev, xMin, xMax, xTol,
                      fOld, f, tStep, omega, fluxRates);

    //... save updated results
    theLidProc.surfaceDepth = x[SURF];
    theLidProc.paveDepth = x[PAVE];
    theLidProc.soilMoisture = x[SOIL];
    theLidProc.storageDepth = x[STOR];
    theLidProc.exflowintotal = theLidProc.exflowintotal + StorageExfil * tStep;
    for (i = 0; i < MAX_LAYERS; i++)
        theLidProc.oldFluxRates[i] = f[i];
    //... return surface outflow (per unit area) from unit
    LidProc[index] = theLidProc;
}

void lidcalculation(double *ptrh, double *ptrh_input, double *ptrqx_input,
                    double *ptrqy_input, double dt, double dx_input, double rainfall, TLidProc *LidProc)
{
    dx = dx_input;
    dy = dx;
    double lidInflow = 0;
    double RFI = rainfall;
    for (int i = 0; i < M * N; i++)
    {
        ptrh[i] = ptrh_input[i];
        if (LidProc[i].lidType > 0)
        {
            //printf("inputh:%f\tsurfacedepth:%f\t", ptrh_input[i], LidProc[i].surfaceDepth);
            LidProc[i].surfaceDepth = ptrh_input[i] + LidProc[i].h_inLid;
            //printf("qxInflow:%f\tqyInflow:%f\t",ptrqx_input[i],ptrqy_input[i]);
            lidInflow = lid_getInflow(ptrqx_input, ptrqy_input, dt, i, RFI);
            //printf("lidInflow:%f\t",lidInflow);
            lidproc_getSurfaceDepth(LidProc, lidInflow, dt, i);
            // if (LidProc[i].surfaceDepth < LidProc[i].lidPram.surface.thickness)
            // {
            //     LidProc[i].h_inLid = LidProc[i].surfaceDepth;
            //     ptrh[i] = 0.0;
            // }
            // else
            // {
                ptrh[i] = LidProc[i].surfaceDepth;
                LidProc[i].h_inLid = 0.0;
                //printf("hnew:%f\tsurfaceDepth:%f",ptrh[i],LidProc[i].surfaceDepth);
            //}
            //ptrh[i]=LidProc[i].surfaceDepth;
            //printf("hnew:%f\t",ptrh[i]);
        }
        //printf("hnew:%f\t",ptrh[i]);
    }
}

void lid_wrtReport(TLidProc *LidProc, double t)
{
    ostringstream outputtime;
    outputtime.precision(6);
    for (int i = 0; i < M * N; i++)
    {
        if (LidProc[i].lidID > 0)
        {
            outputtime << "/home/jinghuajiang/LidTest_case/output/";
            outputtime << "LID_";
            outputtime << i;
            outputtime << "_";
            outputtime << LidProc[i].lidType;
            outputtime << "_";
            outputtime << t;
            outputtime << "s.txt";
            string filename = outputtime.str();
            FILE *fp = fopen(filename.c_str(), "w");
            if ((fp = fopen(filename.c_str(), "w")) == NULL)
            {
                perror("cannot open output file\n");
            }
            else
            {
                int ret = 0;
                ret = fprintf(fp, "%5d"
                                  " %18.12f "
                                  " %18.12f "
                                  " %18.12f "
                                  " %18.12f ",
                              i, LidProc[i].surfaceDepth,
                              LidProc[i].soilMoisture, LidProc[i].storageDepth, LidProc[i].exflowintotal);
                if (ret < 0)
                {
                    perror("write output file failed\n");
                    fclose(fp);
                }
                fclose(fp);
            }

            //fopen(filename.c_str(), "w");
            //printf("%f\t%f\t%f\t%f\n", LidProc[i].surfaceDepth, LidProc[i].soilMoisture, LidProc[i].storageDepth, LidProc[i].exflowintotal);
            // if (fopen(filename.c_str(), "w") == NULL)
            // {
            //     printf("the file was not opened");
            // }
            // else
            // {
            //
            //     fprintf(ptr_file, "%5d"
            //                       " %18.12f "
            //                       " %18.12f "
            //                       " %18.12f "
            //                       " %18.12f ",
            //             i, LidProc[i].surfaceDepth,
            //             LidProc[i].soilMoisture, LidProc[i].storageDepth, LidProc[i].exflowintotal);
            //     fprintf(ptr_file, "\n");
            // //}
            // fclose(ptr_file);
        }
    }
}
