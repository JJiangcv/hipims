
#include <torch/extension.h>
#include "infil.h"
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <algorithm>

static double Fumax;   // saturated water volume in upper soil zone (m)
static double InfilFactor;        
double recoveryFactor=1.0;

#define   ZERO               1.E-10         // Effective zero value
#define   TRUE               1              // Value for TRUE state
#define   FALSE              0              // Value for FALSE state
#define MIN(x,y) (((x)<=(y)) ? (x) : (y))        /* minimum of x and y    */
#define MAX(x,y) (((x)>=(y)) ? (x) : (y))        /* maximum of x and y  */

void grnampt_initState(TGrnAmpt *infil)
//
//  Input:   infil = ptr. to Green-Ampt infiltration object
//  Output:  none
//  Purpose: initializes state of Green-Ampt infiltration for a subcatchment.
//
{
	if (infil == NULL) return;
	infil->IMD = infil->IMDmax;
	infil->Fu = 0.0;
	infil->F = 0.0;
	infil->Sat = FALSE;
	infil->T = 0.0;
}

double grnampt_getInfil(TGrnAmpt *infil, double tstep, double irate,
						double depth, int modelType)
//
//  Purpose: computes Green-Ampt infiltration for a subcatchment or a storage node.
//
{
	// --- find saturated upper soil zone water volume
	InfilFactor = 1.0;
	Fumax = infil->IMDmax * infil->Lu * sqrt(InfilFactor);  //(5.1.013)

	// --- reduce time until next event
	infil->T -= tstep;

	// --- use different procedures depending on upper soil zone saturation
	if (infil->Sat) return grnampt_getSatInfil(infil, tstep, irate, depth);
	else return grnampt_getUnsatInfil(infil, tstep, irate, depth, modelType);
}

//========================================================================================================//
double grnampt_getUnsatInfil(TGrnAmpt *infil, double tstep, double irate,
	double depth, int modelType)
	//
	//  Input:   infil = ptr. to Green-Ampt infiltration object
	//           tstep =  runoff time step (sec),
	//           irate = net "rainfall" rate to upper zone (ft/sec);
	//                 = rainfall + snowmelt + runon,
	//                   does not include ponded water (added on below)
	//           depth = depth of ponded water (ft)
	//           modelType = either GREEN_AMPT or MOD_GREEN_AMPT
	//  Output:  returns infiltration rate (ft/sec)
	//  Purpose: computes Green-Ampt infiltration when upper soil zone is
	//           unsaturated.
	//
{
	double ia, c1, F2, dF, Fs, kr, ts;
	//double InfilFactor = 1.0; //
	double ks = infil->Ks * InfilFactor;                                       //(5.1.013)
	double lu = infil->Lu * sqrt(InfilFactor);                                 //(5.1.013)

	// --- get available infiltration rate (rainfall + ponded water)
	ia = irate + depth / tstep;
	//printf("ia:%.10f\t",ia);
	if (ia < ZERO) ia = 0.0;

	// --- no rainfall so recover upper zone moisture
	if (ia == 0.0)
	{
		if (infil->Fu <= 0.0) return 0.0;
		kr = lu / 90000.0 / 0.3048 * recoveryFactor;  //s
		dF =  Fumax * kr * tstep; 
		infil->F -= dF;
		infil->Fu -= dF;
		if (infil->Fu <= 0.0)
		{
			infil->Fu = 0.0;
			infil->F = 0.0;
			infil->IMD = infil->IMDmax;
			return 0.0;
		}

		// --- if new wet event begins then reset IMD & F
		if (infil->T <= 0.0)
		{
			infil->IMD = (Fumax - infil->Fu) / lu;
			infil->F = 0.0;
		}
		return 0.0;
	}

	// --- rainfall does not exceed Ksat
	if (ia <= ks)
	{
		dF = ia * tstep;
		infil->F += dF;
		infil->Fu += dF;
		infil->Fu = MIN(infil->Fu, Fumax);
		if (modelType == GREEN_AMPT &&  infil->T <= 0.0)
		{
			infil->IMD = (Fumax - infil->Fu) / lu;
			infil->F = 0.0;
		}
		return ia;
	}

	// --- rainfall exceeds Ksat; renew time to drain upper zone
	infil->T = 5400.0 / (lu*3.281) / recoveryFactor;         

	// --- find volume needed to saturate surface layer
	Fs = ks * (infil->S + depth) * infil->IMD / (ia - ks);
	//printf("ks:%.10f\t infil.s:%.10f\t depth:%.10f\t infilIMD:%.10f\t%l.10\t",ks,infil->S,depth, infil->IMD,ia);
	//printf("calre:%.10f",ks * (infil->S + depth) * infil->IMD);
	//printf("Fs:%.10f\t",Fs);

	// --- surface layer already saturated
	if (infil->F > Fs)
	{
		infil->Sat = TRUE;
		return grnampt_getSatInfil(infil, tstep, irate, depth);
	}

	// --- surface layer remains unsaturated
	if (infil->F + ia*tstep  < Fs)
	{
		dF = ia * tstep;
		infil->F += dF;
		infil->Fu += dF;
		infil->Fu = MIN(infil->Fu, Fumax);
		return ia;
	}

	// --- surface layer becomes saturated during time step;
	// --- compute portion of tstep when saturated
	ts = tstep - (Fs - infil->F) / ia;
	if (ts <= 0.0) ts = 0.0;

	// --- compute new total volume infiltrated
	c1 = (infil->S + depth) * infil->IMD;
	F2 = grnampt_getF2(Fs, c1, ks, ts);
	if (F2 > Fs + ia*ts) F2 = Fs + ia*ts;

	// --- compute infiltration rate
	dF = F2 - infil->F;
	infil->F = F2;
	infil->Fu += dF;
	infil->Fu = MIN(infil->Fu, Fumax);
	infil->Sat = TRUE;
	//printf("unsaturdf:%f\tdt:%f\t",dF,tstep);
	return dF / tstep;
}

double grnampt_getSatInfil(TGrnAmpt *infil, double tstep, double irate,
	double depth)
{
	double ia, c1, dF, F2;
	double ks = infil->Ks * InfilFactor;                                       //(5.1.013)
	double lu = infil->Lu * sqrt(InfilFactor);                                 //(5.1.013)

	// --- get available infiltration rate (rainfall + ponded water)
	ia = irate + depth / tstep;
	if (ia < ZERO) return 0.0;

	// --- re-set new event recovery time
	infil->T = 5400.0 / (lu*3.281) / recoveryFactor;

	// --- solve G-A equation for new cumulative infiltration volume (F2)
	c1 = (infil->S + depth) * infil->IMD;
	F2 = grnampt_getF2(infil->F, c1, ks, tstep);
	//printf("F2:%f\t",F2);
	dF = F2 - infil->F;

	// --- all available water infiltrates -- set saturated state to false
	if (dF > ia * tstep)
	{
		dF = ia * tstep;
		infil->Sat = FALSE;
	}

	// --- update total infiltration and upper zone moisture deficit
	infil->F += dF;
	infil->Fu += dF;
	infil->Fu = MIN(infil->Fu, Fumax);
	//printf("SatuDf:%f\tdt:%f\t",dF,tstep);
	return dF / tstep;
}

double grnampt_getF2(double f1, double c1, double ks, double ts)
//
//  Input:   f1 = old infiltration volume (ft)
//           c1 = head * moisture deficit (ft)
//           ks = sat. hyd. conductivity (ft/sec)
//           ts = time step (sec)
//  Output:  returns infiltration volume at end of time step (ft)
//  Purpose: computes new infiltration volume over a time step
//           using Green-Ampt formula for saturated upper soil zone.
//
{
	int    i;
	double f2 = f1;
	double f2min;
	double df2;
	double c2;

	// --- find min. infil. volume
	f2min = f1 + ks * ts;

	// --- use min. infil. volume for 0 moisture deficit
	if (c1 == 0.0) return f2min;

	// --- use direct form of G-A equation for small time steps
	//     and c1/f1 < 100
	if (ts < 10.0 && f1 > 0.01 * c1)
	{
		f2 = f1 + ks * (1.0 + c1 / f1) * ts;
		return MAX(f2, f2min);
	}

	// --- use Newton-Raphson method to solve integrated G-A equation
	//     (convergence limit reduced from that used in previous releases)
	//printf("f2:%f\tf1:%f\t", f2, f1);
	c2 = c1 * log((f1 + c1)) - ks * ts;
	//printf("c2:%f\t c1:%f\t", c2, c1);
	for (i = 1; i <= 20; i++)
	{
		df2 = (f2 - f1 - c1 * log(f2 + c1) + c2) / (1.0 - c1 / (f2 + c1));
		if (abs(df2) < 0.00001)
		{
			//printf("f2:%f\t f2min:%f\t", f2, f2min);
			return MAX(f2, f2min);
		}
		f2 -= df2;
	}
	return f2min;
}