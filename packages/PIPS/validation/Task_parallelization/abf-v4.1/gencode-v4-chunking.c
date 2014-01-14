// Copyright THALES 2010 All rights reserved
//
// THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING,
// WITHOUT LIMITATION, ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, INNOVATIVE OR RELEVANT NATURE, FITNESS 
// FOR A PARTICULAR PURPOSE OR COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.

/* ------ Def ----- */
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#if defined(__cplusplus) ||defined(_cplusplus)
extern "C" {
#endif
#include "appli_headers.h"
              
#if defined(__cplusplus) ||defined(_cplusplus)
}
#endif
              
              
extern s_para_vol *ptr_vol;
extern s_para_vol_out *ptr_vol_out;


extern u_para_private param_private;
extern void *ptr_vol_void;
extern int size_vol;
extern void *ptr_vol_out_void;
extern int size_vol_out;

/* Define macro for prototyping functions on ANSI & non-ANSI compilers */
#ifndef ARGS
#if defined(__STDC__) || defined(__cplusplus)
#define ARGS(args) args
#else
#define ARGS(args) ()
#endif
#endif

/* Define constants TRUE and FALSE for portability */
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#define GTIME 1
int idxTime = 0;
int firstBloc = 0;



int peId=0; /* Default value for mono-PE host execution */

int nbDimIN = 3;
int nbDimOUT = 3;

int org[3];
int length[3];
int ampl[3];
int no_dim[3];

int sauts_in[3];
int sauts_out[3];

#define AFFECT3(X,V1,V2,V3) {X[0] = V1; X[1] = V2; X[2] = V3;}

  int tab_index[5];
  Cplfloat ValSteer[ATL_NUM_BEAMS][ATL_NUM_TRACKS];
  Cplfloat STIM[19][2000][64]; /*[2432000][1]*/
  Cplfloat CI_out[16][32][1999];
  Cplfloat INV_out[64][64]; /*[4096][1]*/
  Cplfloat DBF_out[18][16][2000]; /*[576000][1]*/
  Cplfloat DOP_out[16][2000][32]; /*[1024000][1]*/
  Cplfloat CI_out[16][32][1999]; /*[1023488][1]*/
  Cplfloat X_2_out[19][64][2000]; /*[2432000][1]*/
  Cplfloat CORR_out[19][64][64]; /*[77824][1]*/
  Cplfloat ADD_CORR_out[64][64]; /*[4096][1]*/
  Cplfloat CTR_out[16][64]; /*[1024][1]*/
  Cplfloat X_4_out[16][2000][18]; /*[576000][1]*/
  Cplfloat X_3_out[16][32][2000]; /*[1024000][1]*/
  Cplfloat X__out[2000][64][19]; /*[2432000][1]*/
  Cplfloat X_5_out[18][2000][64]; /*[2304000][1]*/
  Cplfloat sel_out[19][64][200]; /*[243200][1]*/
  Cplfloat X_6_out[64][19][64]; /*[77824][1]*/
  Cplfloat mti_out[2000][64][18]; /*[2304000][1]*/



/* ------ Functions ----- */
void main_PE0( ){
  int i0,i1,i31,i2,i41,i5,i6,i7,i8,i9,i10,i11,i12, i32,i42,i33,i43, i71,i81, i14,i15,i13, i01,i02;

 
  trigger_1(tab_index); //sel - mode: init rate: 
  trigger_2(ValSteer); //CTR - mode: init rate: 

  // appelle fonction : AST_GEN_STIM 
  GEN_STIM(64,2000,19, STIM, 30, 45, 20);
  turn7(19,2000,64,STIM,X__out);

  for (i0=0;i0<600;i0++)
    MTI(19,64, X__out[i0], mti_out[i0]);
  for (i01=600;i01<1200;i01++)
    MTI(19,64, X__out[i01], mti_out[i01]);
  for (i02=1200;i02<2000;i02++)
    MTI(19,64, X__out[i02], mti_out[i02]);


  turn3(2000,64,18,mti_out,X_5_out);
  turn4(19,2000,64,STIM,X_2_out);
  for (i71=0;i71<6;i71++)
    for (i81=0;i81<64;i81++)
      SEL(2000, X_2_out[i71][i81], tab_index, 40, sel_out[i71][i81]);

  for (i11=6;i11<12;i11++)
    for (i12=0;i12<64;i12++)
      SEL(2000, X_2_out[i11][i12], tab_index, 40, sel_out[i11][i12]);

  for (i7=12;i7<19;i7++)
    for (i8=0;i8<64;i8++)
      SEL(2000, X_2_out[i7][i8], tab_index, 40, sel_out[i7][i8]);

  for (i1=0;i1<5;i1++)
    COR(200,64, sel_out[i1], CORR_out[i1]);
   //generate a bug in the clustering
  for (i14=5;i14<10;i14++)
    COR(200,64, sel_out[i14], CORR_out[i14]);
  for (i15=10;i15<15;i15++)
    COR(200,64, sel_out[i15], CORR_out[i15]);
  for (i13=15;i13<19;i13++)
    COR(200,64, sel_out[i13], CORR_out[i13]);

  turn6(19,64,64,CORR_out,X_6_out);

  for (i2=0;i2<16;i2++)
    ADD_COR(64, 19, X_6_out[i2], ADD_CORR_out[i2]);
  for (i2=16;i2<32;i2++)
    ADD_COR(64, 19, X_6_out[i2], ADD_CORR_out[i2]);
  for (i2=32;i2<48;i2++)
    ADD_COR(64, 19, X_6_out[i2], ADD_CORR_out[i2]);
  for (i2=48;i2<64;i2++)
    ADD_COR(64, 19, X_6_out[i2], ADD_CORR_out[i2]);

  INV(64, ADD_CORR_out, INV_out);

  Matmat(16, 64, 64,  ValSteer, INV_out, CTR_out);

  for (i31=0;i31<4;i31++)
    for (i41=0;i41<16;i41++)
      Matmat_transp(	1,64,2000,
			CTR_out[i41],
			X_5_out[i31],
			DBF_out[i31][i41]);
  for (i32=4;i32<8;i32++)
    for (i42=0;i42<16;i42++)
      Matmat_transp(	1,64,2000,
			CTR_out[i42],
			X_5_out[i32],
			DBF_out[i32][i42]);
  for (i33=8;i33<12;i33++)
    for (i43=0;i43<16;i43++)
      Matmat_transp(	1,64,2000,
			CTR_out[i43],
			X_5_out[i33],
			DBF_out[i33][i43]);

    for (i9=12;i9<18;i9++)
      for (i10=0;i10<16;i10++)
        Matmat_transp(	1,64,2000,
			CTR_out[i10],
			X_5_out[i9],
			DBF_out[i9][i10]);

  turn7(18,16,2000,DBF_out,X_4_out);

  for (i5=5;i5<16;i5++)
    DOP(18, 2000,X_4_out[i5], 32, DOP_out[i5]);

  turn4(16,2000,32,DOP_out,X_3_out);

  for (i6=0;i6<16;i6++)
    MTI(1999, 32,X_3_out[i6], CI_out[i6]);

}

/************************ Test ***********************************/
/*int testResult(float* tab, char * filename) {
	FILE *fichier;
	fichier = fopen(filename, "r");
	if (fichier) {
		float val1, val2;
		int i = 0;
		while (fscanf(fichier, "%f - %f", &val1, &val2) != EOF) {
			if (fabs(*(tab + i * 2) - val1) > 0.0001) {
				return -1;
			}
			if (fabs(*(tab + i * 2 + 1) - val2) > 0.0001) {
				return -1;
			}
			i++;
		}
		fclose(fichier);
		return 0;
	} else {
		printf("Test impossible !\n");
		return -1;
	}
}

*/
/* ------ Main ----- */
/* main function */
int main(int argc, char* argv[])
{
  struct timeval tvStart,tvEnd;
  double linStart = 0,linEnd = 0,lTime = 0;


  /* triggers at 'init' */
 

    gettimeofday (&tvStart,NULL);

    main_PE0();
    gettimeofday (&tvEnd,NULL);

    linStart = ((double)tvStart.tv_sec * 1000 + (double)(tvStart.tv_usec/1000.0));
    linEnd = ((double)tvEnd.tv_sec * 1000 + (double)(tvEnd.tv_usec/1000.0));
    lTime = linEnd-linStart;
    //    if (testResult(CI_out, "./ci_ref.txt") != 0)
    //      printf("### TEST FAILED !!! ###\n");
    //  else
    //    printf("=> TEST OK !\n");
    
    printf("----------------------------------------------------------------------\n");
    printf("*** Minimum global time  : %.3f (ms)\n",lTime);

#ifdef OUTPUT_FILE
    
#endif

  return 0;

}/* End of main */

/* ------ End Main ----- */
