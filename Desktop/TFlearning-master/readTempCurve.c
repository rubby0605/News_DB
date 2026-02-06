#include "GetPrm.h"

int readTempCurve(char FILENAME[25], int num, int idCur){
int i, num2, disp_on;
char str[10240];
FILE *fid;
disp_on = 0;
fid = fopen(FILENAME, "r");
fscanf(fid, "%s %s", str, str); 
if(disp_on == 1) printf("%s\n", str);
fscanf(fid, "%s %s \n", str, str); 
if(disp_on == 1) printf("%s\n", str);
//fscanf(fid, "%s", str); 
if(disp_on == 1) printf("%s\n", str);
if(fid == NULL){
	perror("Error while opening the file. \n");
	exit(EXIT_FAILURE);
	}
switch (idCur){
case 1 :
	for(i=0;i<num;i++){ 
		fscanf(fid, "%16lf", &mat_t[i]); 
		fscanf(fid, "%16lf", &mat_Temp[i]); 
	}
case 2 :
	for(i=0;i<num;i++){ 
		fscanf(fid, "%16lf", &mat_t[i]); 
		fscanf(fid, "%16lf", &mat_Temp[i]); 
	}
case 3 :
	for(i=0;i<num;i++){ 
		fscanf(fid, "%16lf", &mat_t[i]); 
		fscanf(fid, "%16lf", &mat_Temp[i]); 
	}
case 4 :
	for(i=0;i<num;i++){ 
		fscanf(fid, "%16lf", &mat_t[i]); 
		fscanf(fid, "%16lf", &mat_Temp[i]); 
	}
}
if(disp_on == 1){
	for(i=0;i<num;i++) 
		printf("%d, %d, %16lf, %8lf \n", idCur, i, mat_t[i], mat_Temp[i]);
	}

return 0;
}
