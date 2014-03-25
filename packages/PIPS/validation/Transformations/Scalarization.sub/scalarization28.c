/* The scalarization of iRec[0] and iRec[1] is not applied because
 * iRec[] is a global array.
 *
 * FI: I wonder if the previous result was executed and checked
 */

#include <stdio.h>
//-----------------------------------------------------
//
// Code generated with Faust 0.9.10 (http://faust.grame.fr)
//-----------------------------------------------------
#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif

typedef long double quad;
/* link with  */


int iRec0[2];

int getNumInputs() 	{ return 0; }
int getNumOutputs() 	{ return 1; }

void compute (int count, FAUSTFLOAT input[0][0], FAUSTFLOAT output[1][512])
{
  FAUSTFLOAT* output0 = output[0];

  int i;

  for (i=0; i<count; i++) {
    output0[i]=output[0][i];
  }

  for (i=0; i<count; i++) {
    iRec0[0] = (12 + (5 * iRec0[1]));
    output0[i] = (FAUSTFLOAT)(0.5f * iRec0[0]);
    // post processing
    iRec0[1] = iRec0[0];
  }
}
void main()
{
  int count=512;
  FAUSTFLOAT in[0][0];
  FAUSTFLOAT out[1][count];

  compute(count,in,out);
  printf("%f",out[0][count-1]);

}

