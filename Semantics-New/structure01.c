
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
  float re;
  float im;
} Cplfloat;




void STAP_Mat_Invert(int ntt,int nsa, Cplfloat mat[ntt][nsa][ntt][nsa],
		     Cplfloat matinv[ntt][nsa][ntt][nsa])
{

  double inv[ntt*nsa+1][2*ntt*nsa+1][3];
  float pivot[3],coef[3];
  float re,im;

  int i,i1=0,i2=0,j=0,j1=0,j2=0,k,k1=0,k2=0,l,l1=0,l2=0;

  // MOTIF
  for (k1=1; k1<=1;k1++){
    for(i1=0;i1<ntt;i1++) {
      for(i2=0;i2<nsa;i2++) {
	for(j1=0;j1<ntt;j1++) {
	  for(j2=0;j2<nsa;j2++) {
	    inv[i1*nsa+i2+1][j1*nsa+j2+1][1] = mat[i1][i2][j1][j2].re;
	    inv[i1*nsa+i2+1][j1*nsa+j2+1][2] =  mat[i1][i2][j1][j2].im;
	  }
	}
      }
    }
    for (i =1; i<nsa*ntt+1; i++) {
      for (j=1; j<nsa*ntt+1; j++)
	{
	  if (i==j) {
	    inv[i][nsa*ntt+j][1] =1.0;
	    inv[i][nsa*ntt+j][2] =0.0;
	  }
	  else {
	    inv[i][nsa*ntt+j][1] =0.0;
	    inv[i][nsa*ntt+j][2] =0.0;
	  }
	}
    }
    for(i=1;i<ntt*nsa+1;i++) {

      pivot[1]=inv[i][i][1];
      pivot[2]=inv[i][i][2];

      if(pivot[1] == 0.) {
	printf("\n Pivot nul re = %f , im = %f\n",pivot[1],pivot[2]);
	exit(0);
      }
      for(j=i;j<2*ntt*nsa+1;j++) {
	re = inv[i][j][1];
	im = inv[i][j][2];
	inv[i][j][1] = (re * pivot[1] + im * pivot[2])/(pivot[1] * pivot[1] + pivot[2] * pivot[2]);
	inv[i][j][2] = (im * pivot[1] - re * pivot[2])/(pivot[1] * pivot[1] + pivot[2] * pivot[2]);
      }


      for(k=1;k<ntt*nsa+1;k++) {
	if(i!=k) {
	  coef[1] = inv[k][i][1];
	  coef[2] = inv[k][i][2];

	  for(l=i;l<2*ntt*nsa+1;l++) {
	    inv[k][l][1] -= (coef[1] * inv[i][l][1] - coef[2] * inv[i][l][2]);
	    inv[k][l][2] -= (coef[1] * inv[i][l][2] + coef[2] * inv[i][l][1]);
	  }
	}
      }
    }


    for(i1=0;i1<ntt;i1++) {
      for(i2=0;i2<nsa;i2++) {
	for(j1=0;j1<ntt;j1++) {
	  for(j2=0;j2<nsa;j2++) {
	    matinv[i1][i2][j1][j2].re = (float) inv[i1*nsa+i2+1][j1*nsa+j2+nsa*ntt+1][1];
	    matinv[i1][i2][j1][j2].im = (float) inv[i1*nsa+i2+1][j1*nsa+j2+nsa*ntt+1][2];

	  }
	}
      }
    }

  }
}
