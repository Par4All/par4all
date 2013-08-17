// Copyright � THALES 2010 All rights reserved
//
// THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING,
// WITHOUT LIMITATION, ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, INNOVATIVE OR RELEVANT NATURE, FITNESS 
// FOR A PARTICULAR PURPOSE OR COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.

#ifndef __GPUradar__
#define __GPUradar__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define C 3.E8
#define Pi 3.141592653589793238

typedef struct {
  float   re;
  float   im;
} Cplfloat;

void GEN_STIM(int ant, int rg, int pul, Cplfloat Stim[pul][rg][ant], float dephant, float dephdop, int cdtarg);
void SEL(int cd,  Cplfloat In[cd],int tab_index[5], int nb_per_slot,  Cplfloat Out[5*nb_per_slot]);
void MTI ( int pul, int ant, Cplfloat Vin[ant][pul], Cplfloat Vout[ant][pul-1]);
void COR( int Nb_rg, int Nb_ant,Cplfloat In[Nb_ant][Nb_rg], Cplfloat Out[Nb_ant][Nb_ant]);
void ADD_COR(int Nb_ant, int Nb_pul, Cplfloat In[Nb_pul][Nb_ant], Cplfloat Out[Nb_ant]);
void INV( int nb_cols, Cplfloat Matin[nb_cols][nb_cols], Cplfloat Matout[nb_cols][nb_cols]);
void Matmat( int Nrows1,int Ncols1, int Ncols2, Cplfloat Mat1[Nrows1][Ncols1], Cplfloat Mat2[Ncols1][Ncols2], Cplfloat Matprod[Nrows1][Ncols2]);
void Matmat_transp(int Nrows1, int Ncols1, int Nrows2, Cplfloat Mat1[Ncols1], Cplfloat Mat2[Nrows2][Ncols1], Cplfloat Matprod[Nrows2]);
void DOP(int pul, int rg, Cplfloat in[rg][pul], int Nsup, Cplfloat out[rg][Nsup]);
void turn3(int nsa,int nrec,int dim3, Cplfloat a[nsa][nrec][dim3], Cplfloat b[dim3][nsa][nrec]);
void turn4(int dim1,int dim2,int dim3, Cplfloat a[dim1][dim2][dim3],Cplfloat b[dim1][dim3][dim2]);
void turn6(int dim1,int dim2,int dim3,Cplfloat a[dim1][dim2][dim3],Cplfloat b[dim2][dim1][dim3] );
void turn7(int dim1,int dim2,int dim3,Cplfloat a[dim1][dim2][dim3], Cplfloat b[dim2][dim3][dim1] );

void XShuffleA_CF_3(int in1,
					int in2,
					int in3,
					Cplfloat ptr_in[in1][in2][in3], 
					int sauts_in[3],
					int Org[3],
					int length[3],
					int no_dim[3],
					int ampl[3],
					int sauts_out[3],
					int out1,
					int out2,
					int out3,
					Cplfloat ptr_out[out1][out2][out3]);
						
void writeFile (int size, char* nom, Cplfloat* buffer);						
int testResult(float* tab, char * filename);

#endif
