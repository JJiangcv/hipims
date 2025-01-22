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





TLidProc *LidProc;


char *LidLayerWords[] =
    {"SURFACE", "SOIL", "STORAGE", "PAVEMENT", "DRAINMAT", "DRAIN",
     "REMOVALS", NULL};




//==============================================================================================//



void lid_wrtReport(double lid_t_out, TLidProc *LidProc)
{
  ostringstream outputtime;
	outputtime.precision(6);
  for (int i = 0; i < M*N; i++)
	{
		if (LidProc[i].lidID > 0)
		{
      outputtime << "/home/jinghuajiang/LidTest_case/output/";
      outputtime << "LID_";
			outputtime << i;
			outputtime << "_";
			outputtime << LidProc[i].lidType;
			outputtime << "_";
			outputtime << lid_t_out;
			outputtime << "s.txt";
      string filename = outputtime.str();
			FILE *ptr_file;
      ptr_file = fopen(filename.c_str(),"w");
      fprintf(ptr_file, "%5d" " %18.12f "" %18.12f "" %18.12f "" %18.12f ", i, LidProc[i].surfaceDepth,
        LidProc[i].soilMoisture, LidProc[i].storageDepth, LidProc[i].exflowintotal);
      fprintf(ptr_file, "\n");
      fclose(ptr_file);
    }
  }
}

