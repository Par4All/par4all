#define sizeN 1024
#define sizeM 1024
#define z(i,j) (i)*sizeM+(j)
#include <stdio.h>
#include <stdlib.h>

void main(int argc, char *argv[])
{
  float Gx[sizeN*sizeM], Gy[sizeN*sizeM], Ixx[sizeN*sizeM], Iyy[sizeN*sizeM], Ixy[sizeN*sizeM], Sxx[sizeN*sizeM], Sxy[sizeN*sizeM], Syy[sizeN*sizeM], in[sizeN*sizeM], out[sizeN*sizeM];

#pragma distributed on_cluster=0
  {
    unsigned int i,j;
    for(i = 0 ; i < sizeN ; i++)
      for(j = 0 ; j < sizeM ; j=j+3){
        in[z(i,j)] = 5; /* A dummy initialisation, we could load a true image instead */
        in[z(i,j+1)] = 3;in[z(i,j+2)] = 7;
      }  
  }
 /* Now we run the Harris procedure */
 
 //Sobel
#pragma distributed on_cluster=03
  {
    unsigned int i,j;
    for(i = 1; i < sizeN - 1; i++)
      for(j = 1; j < sizeM - 1; j++){
        Gx[z(i,j)] = (in[z(i-1,j-1)]-in[z(i-1,j+1)]+2*(in[z(i,j-1)]-in[z(i,j+1)])+in[z(i+1,j-1)]-in[z(i+1,j+1)])/8;
      }
  }
#pragma distributed on_cluster=1
  {
    unsigned int i,j;
    for(i = 1; i < sizeN - 1; i++)
      for(j = 1; j < sizeM - 1; j++){
        Gy[z(i,j)] = (in[z(i-1,j-1)]-in[z(i+1,j-1)]+2*(in[z(i-1,j)]-in[z(i+1,j)])+in[z(i-1,j+1)]+in[z(i+1,j+1)])/8;
      }
  }
 
 //Multiply
#pragma distributed on_cluster=0
  {
    unsigned int i,j;
    for(i = 0; i < sizeN; i++)
    for(j = 0; j < sizeM; j++)
      Ixx[z(i,j)] = Gx[z(i,j)]*Gx[z(i,j)];
  }
#pragma distributed on_cluster=1
  {
    unsigned int i,j;
    for(i = 0; i < sizeN; i++)
    for(j = 0; j < sizeM; j++)
      Iyy[z(i,j)] = Gy[z(i,j)]*Gy[z(i,j)];
  }
#pragma distributed on_cluster=2
  {
    unsigned int i,j;
    for(i = 0; i < sizeN; i++)
    for(j = 0; j < sizeM; j++)
      Ixy[z(i,j)] = Gx[z(i,j)]*Gy[z(i,j)];
  }
 
 //Gauss
#pragma distributed on_cluster=0
  {
    unsigned int i,j;
    for(i = 1; i < sizeN - 1; i++)
    for(j = 1; j < sizeM - 1; j++)
      Sxx[z(i,j)] = (Ixx[z(i-1,j-1)]+2*Ixx[z(i-1,j)]+Ixx[z(i-1,j+1)]+2*(Ixx[z(i,j-1)]+2*Ixx[z(i,j)]+Ixx[z(i,j+1)])+Ixx[z(i+1,j-1)]+Ixx[z(i+1,j)]+Ixx[z(i+1,j+1)])/16;
  }
#pragma distributed on_cluster=1
  {
    unsigned int i,j;
    for(i = 1; i < sizeN - 1; i++)
    for(j = 1; j < sizeM - 1; j++)
      Syy[z(i,j)] = (Iyy[z(i-1,j-1)]+2*Iyy[z(i-1,j)]+Iyy[z(i-1,j+1)]+2*(Iyy[z(i,j-1)]+2*Iyy[z(i,j)]+Iyy[z(i,j+1)])+Iyy[z(i+1,j-1)]+Iyy[z(i+1,j)]+Iyy[z(i+1,j+1)])/16;
  }
#pragma distributed on_cluster=2
  {
    unsigned int i,j;
    for(i = 1; i < sizeN - 1; i++)
    for(j = 1; j < sizeM - 1; j++)
      Sxy[z(i,j)] = (Ixy[z(i-1,j-1)]+2*Ixy[z(i-1,j)]+Ixy[z(i-1,j+1)]+2*(Ixy[z(i,j-1)]+2*Ixy[z(i,j)]+Ixy[z(i,j+1)])+Ixy[z(i+1,j-1)]+Ixy[z(i+1,j)]+Ixy[z(i+1,j+1)])/16;
  }
 
 //Coarsity
#pragma distributed on_cluster=0
  {
    unsigned int i,j;
    for(i = 0; i < sizeN; i++)
    for(j = 0; j < sizeM; j++)
      out[z(i,j)] = Sxx[z(i,j)]*Syy[z(i,j)]-Sxy[z(i,j)]*Sxy[z(i,j)];
  }
}



