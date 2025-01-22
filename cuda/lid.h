// lid.h
// Date: 26/12/2020
// 

#ifndef LID_H
#define LID_H

#include "infil.h"

#define MAX_LAYERS 4

enum LidLayerTypes {
	SURF,                    // surface layer
	SOIL,                    // soil layer
	STOR,                    // storage layer
	PAVE,                    // pavement layer
	DRAIN					 // underdrain system
};

// LID Surface Layer
typedef struct
{
	double	thickness;			// depression storage height (m)
	double	voidFrac;			// available fraction of storage volume
	double	roughness;			// surface Mannings n
	double	surfSlope;			// land surface slope (fraction)
	double	sideSlope;          // swale side slope (run/rise)
	double    alpha;            // slope/roughness term in Manning eqn.
} TSurfaceLayer;

// LID Pavement Layer
typedef struct
{
	double   thickness;           // layer thickness (ft)
	double   voidFrac;            // void volume / total volume
	double   impervFrac;          // impervious area fraction
	double   kSat;                // permeability (ft/sec)
	double   clogFactor;          // clogging factor
	double   regenDays;           // clogging regeneration interval (days)     //(5.1.013)
	double   regenDegree;         // degree of clogging regeneration           //
}  TPavementLayer;

// LID Soil Layer
typedef struct
{
	double	thickness;			// layer thickness (m)
	double	porosity;			// void volume / total volume
	double	fieldCap;			// field capacity
	double	wiltPoint;			// wilting point
	double	kSat;				// saturated hydraulic conductivity (m/s)
	double	kSlope;				// slope of log(K) v. moisture content curve
	double	suction;			// suction head at wetting fromt (m)
} TSoilLayer;

// LID Storage Layer
typedef struct
{
	double	thickness;			// layer thickness (m)
	double	voidFrac;				// void volume / total volume
	double	kSat;				// saturated hydraulic conductivity (m/s)
	double	clogFactor;			// clogging factor
} TStorageLayer;

// Underdrain System (part of Storage Layer)
typedef struct
{
	double    coeff;              // underdrain flow coeff. (in/hr or mm/hr)
	double    expon;              // underdrain head exponent (for in or mm)
	double    offset;             // offset height of underdrain (ft)
	double    delay;              // rain barrel drain delay time (sec)
	double    hOpen;              // head when drain opens (ft)                //(5.1.013)
	double    hClose;             // head when drain closes (ft)               //
	int       qCurve;             // curve controlling flow rate (optional)    //
}  TDrainLayer;

// Drainage Mat Layer (for green roofs)
typedef struct
{
	double    thickness;          // layer thickness (ft)
	double    voidFrac;           // void volume / total volume
	double    roughness;          // Mannings n for green roof drainage mats
	double    alpha;              // slope/roughness term in Manning equation
}  TDrainMatLayer;

typedef struct
{
	TSurfaceLayer		surface;
	TPavementLayer		pavement;      // pavement layer parameters
	TSoilLayer			soil;
	TStorageLayer		storage;
	TDrainLayer			drain;         // underdrain system parameters
	TDrainMatLayer		drainMat;      // drainage mat layer
} TLidPram;

// LID process - generic LID design per unit of area
typedef class {
public:
	int lidType;  // -1: No lid; 1: bio-cell; 2: rain garden; 3: green roof
				  // 4: infiltration trench; 5: porous pavement
				  // 6: rain barrel; 7: vegetative swale 8:ROOF_DISCON

	int lidID;
	TLidPram lidPram;		 // parameters of different lid;
	TGrnAmpt soilInfil;		 // infil. object for biocell soil layer
	double   surfaceDepth;   // depth of ponded water on surface layer (ft)
	double   paveDepth;      // depth of water in porous pavement layer
	double   soilMoisture;   // moisture content of biocell soil layer
	double   storageDepth;   // depth of water in storage layer (ft)
	double   initSat;        // initial saturation of soil & storage layers
	double		exflowintotal;
	// net inflow - outflow from previous time step for each LID layer (ft/s)
	double   oldFluxRates[MAX_LAYERS];

	double   dryTime;        // time since last rainfall (sec)
	double   oldDrainFlow;   // previous drain flow (cfs)
	double   newDrainFlow;   // current drain flow (cfs)
	double   volTreated;     // total volume treated (ft)                      //(5.1.013)
	double   nextRegenDay;   // next day when unit regenerated                 //

	double     NativeInfil;         // native soil infil. rate (ft/s)
	double     MaxNativeInfil;      // native soil infil. rate limit (ft/s)
} TLidProc;


// Data from HiPIMS
typedef class {
public:
	double h;
	double x;
	double y;
	double qx;
	double qy;
	double u;
	double v;
	double zb;
} Cell;


//--------------------------------------------------------------
//void input_readData(Cell* cell, TLidProc* LidProc, int M, int N, double dx, double dy);
//
//void lid_initialize(Cell* cell, TLidProc* LidProc, int M, int N);
//
//void ini_lidProcPram(TLidProc* LidProc, int i);
//
//void lid_readProcParams(TLidProc* theLidProc, int M, int N);
//
//void lid_updateParams(Cell* cell, TLidProc* LidProc, int M, int N);
//
////void input_readCellData(double t);
//
//double lid_getInflow(Cell grid, double dx, double dy);
//
//void lidproc_getSurdepth(TLidProc* LidProc, double inflow, double tStep, int index);
//
//void lid_wrtReport(TLidProc* LidProc, int M, int N, double t);
//
//void Dom_wrtReport(Cell* cell, TLidProc* LidProc, int M, int N, double t);

#endif //LID_H
