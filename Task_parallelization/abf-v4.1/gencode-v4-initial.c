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
#include <omp.h>

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
  int i0,i1;

 
 trigger_1(tab_index); /*sel - mode: init rate: */
  trigger_2(ValSteer); /*CTR - mode: init rate: */	 	 
  /* appelle fonction : AST_GEN_STIM */
  GEN_STIM(64,2000,19, STIM, 30, 45, 20);

  turn7(19,2000,64,STIM,X__out);

  for (i0=0;i0<2000;i0++)
    MTI(19,64, X__out[i0], mti_out[i0]);

  turn3(2000,64,18,mti_out,X_5_out);

  turn4(19,2000,64,STIM,X_2_out);
  for (i0=0;i0<19;i0++)
    for (i1=0;i1<64;i1++)
      SEL(2000, X_2_out[i0][i1], tab_index, 40, sel_out[i0][i1]);


  for (i0=0;i0<19;i0++)
    COR(200,64, sel_out[i0], CORR_out[i0]);

  turn6(19,64,64,CORR_out,X_6_out);

  for (i0=0;i0<64;i0++)
    ADD_COR(64, 19, X_6_out[i0], ADD_CORR_out[i0]);

  INV(64, ADD_CORR_out, INV_out);

  Matmat(16, 64, 64,  ValSteer, INV_out, CTR_out);
  for (i0=0;i0<18;i0++)
    for (i1=0;i1<16;i1++)
      Matmat_transp(	1,64,2000,
			CTR_out[i1],
			X_5_out[i0],
			DBF_out[i0][i1]);

  turn7(18,16,2000,DBF_out,X_4_out);

  for (i0=0;i0<16;i0++)
    DOP(18, 2000,X_4_out[i0], 32, DOP_out[i0]);

  turn4(16,2000,32,DOP_out,X_3_out);

  for (i0=0;i0<16;i0++)
    MTI(1999, 32,X_3_out[i0], CI_out[i0]);

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
