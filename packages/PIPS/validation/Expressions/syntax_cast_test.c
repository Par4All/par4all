/*
!
! Example of syntax_cast case that has made
!
! to reach the function  same_syntax_name_p() updated for C
!
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
	float  re;
	float  im;
} Cplfloat;

#define Lchirp 16
Cplfloat A[128][128][(int)128];
Cplfloat B[64];

void func2(char* filename, Cplfloat Val[185][7][32] ){
	FILE *fp;
	int i,j;
	int a,b,c;
	static char* saveName;
	char ValwithDot[100];
	double Z;
	char *dot;
	if (saveName != filename) {
		saveName = filename;
	}
	fp= fopen(filename, "w");

	for ( i=0;i<32;i++){
		for(j=0;j<185;j++){
			Z = sqrt( Val[i][j][i].re*  Val[i][j][i].re +  Val[i][j][i].im*Val[i][j][i].im);
					sprintf(ValwithDot,"%e\t",Z);
					dot= strchr(ValwithDot,'.');
					if(dot!=NULL){
						strncpy (dot,",",1);
					}
					fprintf(fp, "%s", ValwithDot);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

void func1(  Cplfloat Chirp[4*Lchirp],  float Kchirp)
{
	int t;
	for(t=0;t<64;t++){
		Chirp[t].re= 0.;
		Chirp[t].im= 0.;
		}

}
int main()
{
	func1( B, 0.2);
	A[0][0][0].re=5.;
	A[0][0][0].im=5.;
	func2("SortiesAdapDop.txt",     A);
	return 0;
}
