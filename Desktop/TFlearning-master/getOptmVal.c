#include"GetPrm.h"

double getOptmVal(double X[7], double tt){
double Res;

Res = X[0] + X[1]*exp(-tt/ X[4]) + X[2]*exp(-tt/ X[5]) + X[3]*exp(-tt/ X[6]);
//printf("Res is %f\n", X[0]+X[1]*exp(-tt/X[4]));

return Res;
}
