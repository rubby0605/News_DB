#include<stdio.h>
#include<stdlib.h>
#include<math.h>

//extern double mat_v1[], mat_v2[];/* the length of these two vector might change.*/
#define numGrid 999 
#define numData 180
#define FILEPATH "input_data/mapDataOR100/C1.txt"
extern double mat_t[numGrid];
extern double mat_Temp[numGrid];
extern double X[7];
//extern double Vini1, Vini2;
//extern double mass, h, Vend;

int readTempCurve(char FILENAME[], int num, int idCur);
double getOptmVal(double X[3], double tt);
double getTFV(double *X, double *xbin, int **gene, double *obj, double *constr);

