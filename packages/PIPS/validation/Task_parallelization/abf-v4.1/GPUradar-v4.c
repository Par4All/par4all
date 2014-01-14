// Copyright � THALES 2010 All rights reserved
//
// THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING,
// WITHOUT LIMITATION, ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, INNOVATIVE OR RELEVANT NATURE, FITNESS 
// FOR A PARTICULAR PURPOSE OR COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.

#include "GPUradar.h"
#include "appli_headers.h"
#include "stdio.h"

Cplfloat conjTransp(Cplfloat a) {
	Cplfloat r;
	r.re = -(a).im;
	r.im = a.re;
	return r;
}

Cplfloat complexMul(Cplfloat a, Cplfloat b) {
	Cplfloat ret;
	ret.re = -(a).im * (b).im + (a).re * (b).re;
	ret.im = (a).im * (b).re + (a).re * (b).im;
	return ret;
}

#define fftKernel2(a,dir) \
{ \
    Cplfloat c; \
	c.re= (a)[0].re;    \
	c.im= (a)[0].im;    \
    (a)[0].re = c.re + (a)[1].re;  \
    (a)[0].im = c.im + (a)[1].im;  \
    (a)[1].re = c.re - (a)[1].re;  \
    (a)[1].im = c.im - (a)[1].im;  \
}

#define fftKernel2S(d1,d2,dir) \
{ \
    Cplfloat c; \
    c.re = (d1).re;   \
    c.im = (d1).im;   \
    (d1).re = c.re + (d2).re;   \
    (d1).im = c.im + (d2).im;   \
    (d2).re = c.re - (d2).re;   \
    (d2).im = c.im - (d2).im;   \
}

#define fftKernel4(a,dir) \
{ \
    fftKernel2S((a)[0], (a)[2], dir); \
    fftKernel2S((a)[1], (a)[3], dir); \
    fftKernel2S((a)[0], (a)[1], dir); \
    Cplfloat tmp; \
	tmp = conjTransp((a)[3]); \
    (a)[3].re = (dir)*tmp.re; \
    (a)[3].im = (dir)*tmp.im; \
    fftKernel2S((a)[2], (a)[3], dir); \
    Cplfloat c; \
	c.re = (a)[1].re; \
	c.im = (a)[1].im; \
    (a)[1].re = (a)[2].re; \
    (a)[1].im = (a)[2].im; \
    (a)[2].re = c.re; \
    (a)[2].im = c.im; \
}

#define fftKernel4s(a0,a1,a2,a3,dir) \
{ \
    fftKernel2S((a0), (a2), dir); \
    fftKernel2S((a1), (a3), dir); \
    fftKernel2S((a0), (a1), dir); \
    Cplfloat tmp; \
	tmp = conjTransp((a3)); \
    (a3).re = (dir)*tmp.re; \
    (a3).im = (dir)*tmp.im; \
    fftKernel2S((a2), (a3), dir); \
    Cplfloat c; \
    c.re = (a1).re; \
    c.im = (a1).im; \
    (a1).im = (a2).im; \
    (a1).re = (a2).re; \
    (a2).re = c.re; \
    (a2).im = c.im; \
}

#define bitreverse8(a) \
{ \
    Cplfloat c; \
    c.re = (a)[1].re; \
    c.im = (a)[1].im; \
    (a)[1].re = (a)[4].re; \
    (a)[1].im = (a)[4].im; \
    (a)[4].re = c.re; \
    (a)[4].im = c.im; \
    c.re = (a)[3].re; \
    c.im = (a)[3].im; \
    (a)[3].re = (a)[6].re; \
    (a)[3].im = (a)[6].im; \
    (a)[6].re = c.re; \
    (a)[6].im = c.im; \
}

#define fftKernel8(a,dir) \
{ \
    Cplfloat w1; \
    Cplfloat tmp; \
	w1.re = 0x1.6a09e6p-1f; \
	w1.im = dir*0x1.6a09e6p-1f;  \
    Cplfloat w3; \
	w3.re = -0x1.6a09e6p-1f; \
	w3.im = dir*0x1.6a09e6p-1f;  \
    Cplfloat c; \
    fftKernel2S((a)[0], (a)[4], dir); \
    fftKernel2S((a)[1], (a)[5], dir); \
    fftKernel2S((a)[2], (a)[6], dir); \
    fftKernel2S((a)[3], (a)[7], dir); \
    (a)[5] = complexMul(w1, (a)[5]); \
	tmp= conjTransp((a)[6]); \
    (a)[6].re = (dir)*(tmp.re); \
    (a)[6].im = (dir)*(tmp.im); \
    (a)[7] = complexMul(w3, (a)[7]); \
    fftKernel2S((a)[0], (a)[2], dir); \
    fftKernel2S((a)[1], (a)[3], dir); \
    fftKernel2S((a)[4], (a)[6], dir); \
    fftKernel2S((a)[5], (a)[7], dir); \
	tmp= conjTransp((a)[3]); \
    (a)[3].re = (dir)*(tmp.re); \
    (a)[3].im = (dir)*(tmp.im); \
	tmp= conjTransp((a)[7]); \
    (a)[7].re = (dir)*(tmp.re); \
    (a)[7].im = (dir)*(tmp.im); \
    fftKernel2S((a)[0], (a)[1], dir); \
    fftKernel2S((a)[2], (a)[3], dir); \
    fftKernel2S((a)[4], (a)[5], dir); \
    fftKernel2S((a)[6], (a)[7], dir); \
    bitreverse8((a)); \
}

#define bitreverse4x4(a) \
{ \
    Cplfloat c; \
    c.re = (a)[1].re; \
    c.im = (a)[1].im; \
	(a)[1].re  = (a)[4].re; \
	(a)[1].im  = (a)[4].im; \
	(a)[4].re  = c.re; \
	(a)[4].im  = c.im;  \
    c.re = (a)[2].re;   \
    c.im = (a)[2].im;   \
	(a)[2].re  = (a)[8].re; \
	(a)[2].im  = (a)[8].im; \
	(a)[8].re  = c.re; \
	(a)[8].im  = c.im;  \
    c.re = (a)[3].re;   \
    c.im = (a)[3].im;   \
	(a)[3].re  = (a)[12].re; \
	(a)[3].im  = (a)[12].im; \
	(a)[12].re = c.re; \
	(a)[12].im = c.im;  \
    c.re = (a)[6].re;   \
    c.im = (a)[6].im;   \
	(a)[6].re  = (a)[9].re; \
	(a)[6].im  = (a)[9].im; \
	(a)[9].re  = c.re; \
	(a)[9].im  = c.im;  \
    c.re = (a)[7].re;   \
    c.im = (a)[7].im;   \
	(a)[7].re  = (a)[13].re; \
	(a)[7].im  = (a)[13].im; \
	(a)[13].re = c.re; \
	(a)[13].im = c.im;  \
    c.re = (a)[11].re;  \
    c.im = (a)[11].im;  \
	(a)[11].re = (a)[14].re; \
	(a)[11].im = (a)[14].im; \
	(a)[14].re = c.re; \
	(a)[14].im = c.im;  \
}

#define fftKernel16(a,dir) \
{ \
    Cplfloat tmp; \
    const float w0 = 0x1.d906bcp-1f; \
    const float w1 = 0x1.87de2ap-2f; \
    const float w2 = 0x1.6a09e6p-1f; \
    fftKernel4s((a)[0], (a)[4], (a)[8],  (a)[12], dir); \
    fftKernel4s((a)[1], (a)[5], (a)[9],  (a)[13], dir); \
    fftKernel4s((a)[2], (a)[6], (a)[10], (a)[14], dir); \
    fftKernel4s((a)[3], (a)[7], (a)[11], (a)[15], dir); \
    tmp.re = w0; \
    tmp.im =  dir*w1; \
    (a)[5]  = complexMul((a)[5], tmp); \
    tmp.re = w2; \
    tmp.im =  dir*w2; \
    (a)[6]  = complexMul((a)[6], tmp); \
    tmp.re = w1; \
    tmp.im =  dir*w0; \
    (a)[7]  = complexMul((a)[7], tmp); \
    tmp.re = w2; \
    tmp.im =  dir*w2; \
    (a)[9]  = complexMul((a)[9], tmp); \
    tmp = conjTransp((a)[10]); \
    (a)[10].re = (dir)*tmp.re; \
    (a)[10].im = (dir)*tmp.im; \
    tmp.re = -w2; \
    tmp.im =  dir*w2; \
    (a)[11] = complexMul((a)[11], tmp); \
    tmp.re = w1; \
    tmp.im =  dir*w0; \
    (a)[13] = complexMul((a)[13], tmp); \
    tmp.re = -w2; \
    tmp.im =  dir*w2; \
    (a)[14] = complexMul((a)[14], tmp); \
    tmp.re = -w0; \
    tmp.im =  dir*-w1; \
    (a)[15] = complexMul((a)[15], tmp); \
    fftKernel4((a), dir); \
    fftKernel4((a) + 4, dir); \
    fftKernel4((a) + 8, dir); \
    fftKernel4((a) + 12, dir); \
    bitreverse4x4((a)); \
}

#define bitreverse32(a) \
{ \
    Cplfloat c1, c2; \
    c1.re = (a)[2].re; \
	c1.im = (a)[2].im; \
 	(a)[2].re = (a)[1].re; \
	(a)[2].im = (a)[1].im; \
	c2.re =  (a)[4].re; \
	c2.im =  (a)[4].im; \
	(a)[4].re = c1.re; \
	(a)[4].im = c1.im; \
	c1.re = (a)[8].re; \
	c1.im = (a)[8].im; \
	(a)[8].re = c2.re; \
	(a)[8].im = c2.im; \
	c2.re =  (a)[16].re; \
	c2.im =  (a)[16].im; \
	(a)[16].re = c1.re; \
	(a)[16].im = c1.im; \
	(a)[1].re = c2.re; \
	(a)[1].im = c2.im; \
    c1.re = (a)[6].re; \
	c1.im = (a)[6].im; \
	(a)[6].re = (a)[3].re; \
	(a)[6].im = (a)[3].im; \
	c2.re =  (a)[12].re; \
	c2.im =  (a)[12].im; \
	(a)[12].re = c1.re; \
	(a)[12].im = c1.im; \
	c1.re = (a)[24].re; \
	c1.im = (a)[24].im; \
	(a)[24].re = c2.re; \
	(a)[24].im = c2.im; \
	c2.re =  (a)[17].re; \
	c2.im =  (a)[17].im; \
	(a)[17].re = c1.re; \
	(a)[17].im = c1.im; \
	(a)[3].re = c2.re; \
	(a)[3].im = c2.im; \
    c1.re = (a)[10].re; \
	c1.im = (a)[10].im; \
	(a)[10].re = (a)[5].re; \
	(a)[10].im = (a)[5].im; \
	c2.re =  (a)[20].re; \
	c2.im =  (a)[20].im; \
	(a)[20].re = c1.re; \
	(a)[20].im = c1.im; \
	c1.re = (a)[9].re; \
	c1.im = (a)[9].im; \
	(a)[9].re = c2.re; \
	(a)[9].im = c2.im; \
	c2.re =  (a)[18].re; \
	c2.im =  (a)[18].im; \
	(a)[18].re = c1.re; \
	(a)[18].im = c1.im; \
	(a)[5].re = c2.re; \
	(a)[5].im = c2.im; \
    c1.re = (a)[14].re; \
	c1.im = (a)[14].im; \
	(a)[14].re = (a)[7].re; \
	(a)[14].im = (a)[7].im; \
	c2.re =  (a)[28].re; \
	c2.im =  (a)[28].im; \
	(a)[28].re = c1.re; \
	(a)[28].im = c1.im; \
	c1.re = (a)[25].re; \
	c1.im = (a)[25].im; \
	(a)[25].re = c2.re; \
	(a)[25].im = c2.im; \
	c2.re =  (a)[19].re; \
	c2.im =  (a)[19].im; \
	(a)[19].re = c1.re; \
	(a)[19].im = c1.im; \
	(a)[7].re = c2.re; \
	(a)[7].im = c2.im; \
    c1.re = (a)[22].re; \
	c1.im = (a)[22].im; \
	(a)[22].re = (a)[11].re; \
	(a)[22].im = (a)[11].im; \
	c2.re =  (a)[13].re; \
	c2.im =  (a)[13].im; \
	(a)[13].re = c1.re; \
	(a)[13].im = c1.im; \
	c1.re = (a)[26].re; \
	c1.im = (a)[26].im; \
	(a)[26].re = c2.re; \
	(a)[26].im = c2.im; \
	c2.re =  (a)[21].re; \
	c2.im =  (a)[21].im; \
	(a)[21].re = c1.re; \
	(a)[21].im = c1.im; \
	(a)[11].re = c2.re; \
	(a)[11].im = c2.im; \
    c1.re = (a)[30].re; \
	c1.im = (a)[30].im; \
	(a)[30].re = (a)[15].re; \
	(a)[30].im = (a)[15].im; \
	c2.re =  (a)[29].re; \
	c2.im =  (a)[29].im; \
	(a)[29].re = c1.re; \
	(a)[29].im = c1.im; \
	c1.re = (a)[27].re; \
	c1.im = (a)[27].im; \
	(a)[27].re = c2.re; \
	(a)[27].im = c2.im; \
	c2.re =  (a)[23].re; \
	c2.im =  (a)[23].im; \
	(a)[23].re = c1.re; \
	(a)[23].im = c1.im; \
	(a)[15].re = c2.re; \
	(a)[15].im = c2.im; \
}

#define fftKernel32(a,dir) \
{ \
    fftKernel2S((a)[0],  (a)[16], dir); \
    fftKernel2S((a)[1],  (a)[17], dir); \
    fftKernel2S((a)[2],  (a)[18], dir); \
    fftKernel2S((a)[3],  (a)[19], dir); \
    fftKernel2S((a)[4],  (a)[20], dir); \
    fftKernel2S((a)[5],  (a)[21], dir); \
    fftKernel2S((a)[6],  (a)[22], dir); \
    fftKernel2S((a)[7],  (a)[23], dir); \
    fftKernel2S((a)[8],  (a)[24], dir); \
    fftKernel2S((a)[9],  (a)[25], dir); \
    fftKernel2S((a)[10], (a)[26], dir); \
    fftKernel2S((a)[11], (a)[27], dir); \
    fftKernel2S((a)[12], (a)[28], dir); \
    fftKernel2S((a)[13], (a)[29], dir); \
    fftKernel2S((a)[14], (a)[30], dir); \
    fftKernel2S((a)[15], (a)[31], dir); \
    Cplfloat tmp; \
    tmp.re = 0x1.f6297cp-1f; \
    tmp.im =  dir*0x1.8f8b84p-3f; \
    (a)[17] = complexMul((a)[17], tmp); \
    tmp.re = 0x1.d906bcp-1f; \
    tmp.im =  dir*0x1.87de2ap-2f; \
    (a)[18] = complexMul((a)[18], tmp); \
    tmp.re = 0x1.a9b662p-1f; \
    tmp.im =  dir*0x1.1c73b4p-1f; \
    (a)[19] = complexMul((a)[19], tmp); \
    tmp.re = 0x1.6a09e6p-1f; \
    tmp.im =  dir*0x1.6a09e6p-1f; \
    (a)[20] = complexMul((a)[20], tmp); \
    tmp.re = 0x1.1c73b4p-1f; \
    tmp.im =  dir*0x1.a9b662p-1f; \
    (a)[21] = complexMul((a)[21], tmp); \
    tmp.re = 0x1.87de2ap-2f; \
    tmp.im =  dir*0x1.d906bcp-1f; \
    (a)[22] = complexMul((a)[22], tmp); \
    tmp.re = 0x1.8f8b84p-3f; \
    tmp.im =  dir*0x1.f6297cp-1f; \
    (a)[23] = complexMul((a)[23], tmp); \
    tmp.re = 0x0p+0f; \
    tmp.im =  dir*0x1p+0f; \
    (a)[24] = complexMul((a)[24], tmp); \
    tmp.re = -0x1.8f8b84p-3f; \
    tmp.im =  dir*0x1.f6297cp-1f; \
    (a)[25] = complexMul((a)[25], tmp); \
    tmp.re = -0x1.87de2ap-2f; \
    tmp.im =  dir*0x1.d906bcp-1f; \
    (a)[26] = complexMul((a)[26], tmp); \
    tmp.re = -0x1.1c73b4p-1f; \
    tmp.im =  dir*0x1.a9b662p-1f; \
    (a)[27] = complexMul((a)[27], tmp); \
    tmp.re = -0x1.6a09e6p-1f; \
    tmp.im =  dir*0x1.6a09e6p-1f; \
    (a)[28] = complexMul((a)[28], tmp); \
    tmp.re = -0x1.a9b662p-1f; \
    tmp.im =  dir*0x1.1c73b4p-1f; \
    (a)[29] = complexMul((a)[29], tmp); \
    tmp.re = -0x1.d906bcp-1f; \
    tmp.im =  dir*0x1.87de2ap-2f; \
    (a)[30] = complexMul((a)[30], tmp); \
    tmp.re = -0x1.f6297cp-1f; \
    tmp.im =  dir*0x1.8f8b84p-3f; \
    (a)[31] = complexMul((a)[31], tmp); \
    fftKernel16((a), dir); \
    fftKernel16((a) + 16, dir); \
    bitreverse32((a)); \
}
// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------
void GEN_STIM(int ant, int rg, int pul, Cplfloat Stim[pul][rg][ant], float dephant, float dephdop, int cdtarg) {
  
  int a, c, p;
  float XR,XI;
  float ant_rd, dop_rd;
  ant_rd = dephant* Pi/180.;
  dop_rd = dephdop*Pi/180.;
  for (p=0; p<pul; p++) {
    for (c=0; c<rg; c++) {
      for (a=0; a<ant; a++) {
	// MOTIF
	XR=0.;
	XI=0.;
	if (c==cdtarg) {
	  XR = cos(ant_rd*a + p*dop_rd);
	  XI = sin(ant_rd*a + p*dop_rd);
	}
	Stim[p][c][a].re = XR;
	Stim[p][c][a].im = XI;
      }
    }
  }
}

void turn7(int dim1,int dim2,int dim3,Cplfloat a[dim1][dim2][dim3], Cplfloat b[dim2][dim3][dim1] )
{
  int i,j,k;
  
	//    MOTIF
  for(i=0;i<dim1;i++) {
    for(j=0;j<dim2;j++) {
      for(k=0;k<dim3;k++) {
	b[j][k][i].re = a[i][j][k].re ;
	b[j][k][i].im = a[i][j][k].im ;
      }
    }
  }
}
// ------------------------------------------------------------------------------------
void MTI ( int pul, int ant, Cplfloat Vin[ant][pul], Cplfloat Vout[ant][pul-1])
{
  int t,p;
  for (p=0; p<ant; p++) {
    for (t=1; t<pul; t++) {
      //MOTIF
      Vout[p][t-1].re= Vin[p][t].re - Vin[p][t-1].re;
      Vout[p][t-1].im= Vin[p][t].im - Vin[p][t-1].im;
    }
  }
}

// ------------------------------------------------------------------------------------
void SEL(int cd, Cplfloat In[cd], int tab_index[5],
	 int nb_per_slot, Cplfloat Out[5*nb_per_slot]) {
  int z, n, t;
  t=0;
  for (z=0; z<5; z++) {
    for (n=0; n<nb_per_slot; n++) {
      //MOTIF
      t=z*nb_per_slot+n;
      Out[t].re= In[tab_index[z]+n].re;
      Out[t].im= In[tab_index[z]+n].im;
    }
  }
}

// ------------------------------------------------------------------------------------
// covariance estimation based on a part of the received signal, averaged on Nb_rg range gates and nb_pul pulses
void COR(int Nb_rg, int Nb_ant, Cplfloat In[Nb_ant][Nb_rg], Cplfloat Out[Nb_ant][Nb_ant]) {
  int rg, a0, a1;
  float SR,SI;
  for (a0 = 0; a0 < Nb_ant; a0++) {
    for (a1 = 0; a1 < Nb_ant; a1++) {
      // MOTIF
      SR = 0.;
      SI = 0.;
      for (rg = 0; rg < Nb_rg; rg++) {
	SR += In[a0][rg].re * In[a1][rg].re + In[a0][rg].im * In[a1][rg].im;
	SI += In[a0][rg].im * In[a1][rg].re - In[a0][rg].re * In[a1][rg].im;
      }
      Out[a0][a1].re = SR;
      Out[a0][a1].im = SI;
    }
  }
}


// ------------------------------------------------------------------------------------
void ADD_COR(int Nb_ant, int Nb_pul, Cplfloat In[Nb_pul][Nb_ant],
	     Cplfloat Out[Nb_ant]) {
  int pul, a0;
  Cplfloat S;
  for (a0=0; a0<Nb_ant; a0++) {
    //MOTIF
    S.re=0.;
    S.im=0.;
    for (pul=0; pul<Nb_pul; pul++) {
      S.re += In[pul][a0].re;
      S.im += In[pul][a0].im;
    }
    Out[a0].re = S.re;
    Out[a0].im = S.im;
  }
}

// ------------------------------------------------------------------------------------
void INV(int nb_cols, Cplfloat Matin[nb_cols][nb_cols],
	 Cplfloat Matout[nb_cols][nb_cols]) {
  int i, j;
  int b=0;
  for (i=0; i<nb_cols; i++) {
    for (j=0; j<nb_cols; j++) {
      //MOTIF
      if (Matin[i][j].re == 1.)
	b =1;
      Matout[i][j].re = 1.;
      if (i!=j)
	Matout[i][j].re = 0.;
      Matout[i][j].im = 0.;
      
    }
  }
}

// ------------------------------------------------------------------------------------
void Matmat(int Nrows1, int Ncols1, int Ncols2, Cplfloat Mat1[Nrows1][Ncols1],
		Cplfloat Mat2[Ncols1][Ncols2], Cplfloat Matprod[Nrows1][Ncols2])

{
  int i, j, k;
  Cplfloat Z;
  
  for (i=0; i<Nrows1; i++) {
    for (j=0; j<Ncols2; j++) {
      //MOTIF
      Z.re=0.;
      Z.im=0.;
      for (k=0; k<Ncols1; k++) {
	Z.re += Mat1[i][k].re * Mat2[k][j].re -Mat1[i][k].im
	  * Mat2[k][j].im;
	Z.im += Mat1[i][k].re * Mat2[k][j].im + Mat1[i][k].im
	  * Mat2[k][j].re;
      }
      
      Matprod[i][j].re=(Z.re );
      Matprod[i][j].im=Z.im;
    }
  }
}
/**
 * .ApplicationModel.CTR - mode: init rate: *
 **/

void trigger_2(Cplfloat ValSteer[16][64]) {
	int i, j;

	// initialiser tab de steering vectors
	//MOTIF
	for (i=0; i< 16; i++) {
		for (j=0; j< 64; j++) {
			ValSteer[i][j].re= cos(j*Pi*(i -8)/512);
			ValSteer[i][j].im= -sin(j*Pi*(i -8)/512);
		}
	}
}
// ------------------------------------------------------------------------------------
void Matmat_transp(int Nrows1, int Ncols1, int Nrows2,
		   Cplfloat Mat1[Ncols1], Cplfloat Mat2[Nrows2][Ncols1],
		   Cplfloat Matprod[Nrows2])
  
{
  int j, k;
  Cplfloat Z;
  
  for (j=0; j<Nrows2; j++) {
    //MOTIF
    Z.re=0.;
    Z.im=0.;
    for (k=0; k<Ncols1; k++) {
      Z.re += Mat1[k].re * Mat2[j][k].re -Mat1[k].im
	* Mat2[j][k].im;
      Z.im += Mat1[k].re * Mat2[j][k].im + Mat1[k].im
	* Mat2[j][k].re;
    }
    
    Matprod[j].re=Z.re;
    Matprod[j].im=Z.im;
  }
}

// ------------------------------------------------------------------------------------

void lazy_FFT(int N, Cplfloat Xin[N], int Nsup, Cplfloat Xout[Nsup]) {
  
  Cplfloat T[Nsup];
  int i;
  // MOTIF
  for (i = 0; i < Nsup; i++) {
    if (i < N) {
      T[i].re = Xin[i].re;
      T[i].im = Xin[i].im;
    } else {
      T[i].re = 0.;
      T[i].im = 0.;
    }
  }
  // mere Fourier transform, not Fast
  double phi = 2. * Pi / Nsup;
  Cplfloat Z;
  int m, n;
  for (m = 0; m < Nsup; m++) {
    Z.re = 0.;
    Z.im = 0.;
    for (n = 0; n < Nsup; n++) {
      Z.re += T[n].re * cos(m * n * phi) + T[n].im * sin(m * n * phi);
      Z.im += -T[n].re * sin(m * n * phi) + T[n].im * cos(m * n * phi);
      
    }
    Xout[m].re = Z.re;
    Xout[m].im = Z.im;
  }
}




void DOP(int pul, int rg, Cplfloat in[rg][pul], int Nsup, Cplfloat out[rg][Nsup]) {
  
  int k;
  //MOTIF
  
  /* Calculate the number m such as 2^m = pulses number */
  int m = ceil(log(pul) / log(2));
  
  /* Calculate 2^m */
  int pow_2_sup = 1 << m;
  for (k=0; k<rg; k++) {
    lazy_FFT(pul, in[k], pow_2_sup, out[k]);
  }
}



void writeFile (int size, char* nom, Cplfloat* buffer)
{
  FILE* file = fopen(nom,"w");
  
  int i = 0 ;
  for (i = 0 ; i < size ; i++)
    {
      fprintf(file,"%f - %f\n", buffer[i].re, buffer[i].im);
    }
  
  fclose(file);
}



void turn3(int nsa,int nrec,int dim3, Cplfloat a[nsa][nrec][dim3], Cplfloat b[dim3][nsa][nrec])
{  int j,k,l;
  
  //    MOTIF
  for(j=0;j<nsa;j++) {
    for(k=0;k<nrec;k++) {
      for(l=0;l<dim3;l++) {
	b[l][j][k].re = a[j][k][l].re ;
	b[l][j][k].im = a[j][k][l].im ;
      }
    }
  }
}

void turn4(int dim1,int dim2,int dim3, Cplfloat a[dim1][dim2][dim3],Cplfloat b[dim1][dim3][dim2])
{  int j,k,l;
  
	//    MOTIF
  for(j=0;j<dim1;j++) {
    for(k=0;k<dim2;k++) {
      for(l=0;l<dim3;l++) {
	b[j][l][k].re = a[j][k][l].re ;
	b[j][l][k].im = a[j][k][l].im ;
      }
    }
  }
}


void turn6(int dim1,int dim2,int dim3,Cplfloat a[dim1][dim2][dim3],Cplfloat b[dim2][dim1][dim3] )
{
  int i,j,k;

	//    MOTIF
  for(i=0;i<dim1;i++) {
    for(j=0;j<dim2;j++) {
      for(k=0;k<dim3;k++) {
	b[j][i][k].re = a[i][j][k].re ;
	b[j][i][k].im = a[i][j][k].im ;
	
      }
    }
  }
}


