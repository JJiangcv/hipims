#ifndef INFIL_H
#define INFIL_H

//---------------------
// Enumerated Constants
//---------------------
enum InfilType {
	HORTON,                      // Horton infiltration
	MOD_HORTON,                  // Modified Horton infiltration
	GREEN_AMPT,                  // Green-Ampt infiltration
	MOD_GREEN_AMPT,              // Modified Green-Ampt infiltration
	CURVE_NUMBER
};               // SCS Curve Number infiltration

//-------------------------
// Green-Ampt Infiltration
//-------------------------
typedef struct
{
	double        S;               // avg. capillary suction (ft)
	double        Ks;              // saturated conductivity (ft/sec)
	double        IMDmax;          // max. soil moisture deficit (ft/ft)
								   //-----------------------------
	double        IMD;             // current initial soil moisture deficit
	double        F;               // current cumulative infiltrated volume (ft)
	double        Fu;              // current upper zone infiltrated volume (ft)
	double        Lu;              // depth of upper soil zone (ft)
	double        T;               // time until start of next rain event (sec)
	char          Sat;             // saturation flag
}  TGrnAmpt;


double  grnampt_getInfil(TGrnAmpt *infil, double tstep, double irate,
	double depth, int modelType);
//
double grnampt_getSatInfil(TGrnAmpt *infil, double tstep, double irate,
	double depth);
//
double grnampt_getUnsatInfil(TGrnAmpt *infil, double tstep, double irate,
	double depth, int modelType);
//
double grnampt_getF2(double f1, double c1, double ks, double ts);
//
void grnampt_initState(TGrnAmpt *infil);

#endif //INFIL_H
