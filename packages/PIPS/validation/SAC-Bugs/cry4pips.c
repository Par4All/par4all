#include <stdio.h>
/* faust-flag*/
//-----------------------------------------------------
//
// Code generated with Faust 0.9.10 (http://faust.grame.fr)
//-----------------------------------------------------
#ifndef FAUSTFLOAT
#define FAUSTFLOAT float
#endif  

typedef long double quad;
/* link with  */

FAUSTFLOAT 	fslider0;
float 	fRec1[2024];
float 	fConst0;
float 	fConst1;
float 	fRec2[2024];
float 	fRec3[2024];
float 	fRec0[2024];
float 	fRec4[2024];
float 	fRec5[2024];
float 	fRec6[2024];
float 	fRec7[2024];
float 	fRec8[2024];
float 	fRec9[2024];
float 	fRec10[2024];
int fSamplingFreq;

int factorielle(int n)
{
    int i; /*compteur de boucle*/
    int maxi ;
    maxi = n-1 ;
    if(n==0)
    {
        return 1 ;
    }
    else
    {
        for(i=1 ; i<=maxi ; i++)
        {
            n=n*(n-i);
        }
        return n;
    }
}
float cosf(float x)
{
    float E = 0.001;
    int c=1;
    int n=-2;
    float cos = 0;
    do {
        cos = c * (pow(x,2+n)/factorielle(2+n)) +cos;
        c=-1*c;
        n=n+2;
    }while ((pow(x,n+1) / factorielle(n+1)) >= E);
    return cos;
} 


int getNumInputs() 	{ return 8; }
int getNumOutputs() 	{ return 8; }
static void classInit(int samplingFreq) 
{
}
void instanceInit(int samplingFreq) {
    int i;		
    fSamplingFreq = samplingFreq;
    fslider0 = 100.0f;
    for (i=0; i<2; i++) fRec1[i] = 0;
    fConst0 = (1413.716694f / fSamplingFreq);
    fConst1 = (2827.433388f / fSamplingFreq);
    for (i=0; i<2; i++) fRec2[i] = 0;
    for (i=0; i<2; i++) fRec3[i] = 0;
    for (i=0; i<3; i++) fRec0[i] = 0;
    for (i=0; i<3; i++) fRec4[i] = 0;
    for (i=0; i<3; i++) fRec5[i] = 0;
    for (i=0; i<3; i++) fRec6[i] = 0;
    for (i=0; i<3; i++) fRec7[i] = 0;
    for (i=0; i<3; i++) fRec8[i] = 0;
    for (i=0; i<3; i++) fRec9[i] = 0;
    for (i=0; i<3; i++) fRec10[i] = 0;
}

void cry4pips (int count, FAUSTFLOAT input[8][1024], FAUSTFLOAT output[8][1024]) 
{
    float 	fSlow0 = fslider0;
    float 	fSlow1 = (1.000000e-04f * pow(4.0f,fSlow0));
    float 	fSlow2 = pow(2.0f,(2.3f * fSlow0));
    float 	fSlow3 = (1 - (fConst0 * (fSlow2 / pow(2.0f,(1.0f + (2.0f * (1.0f - fSlow0)))))));
    float 	fSlow4 = (1.000000e-03f * (0 - (2.0f * (pow(fConst1,fSlow2) * fSlow3))));
    float 	fSlow5 = (1.000000e-03f * (fSlow3 * fSlow3));
    int i;		
    FAUSTFLOAT* input0 = input[0];
    FAUSTFLOAT* input1 = input[1];
    FAUSTFLOAT* input2 = input[2];
    FAUSTFLOAT* input3 = input[3];
    FAUSTFLOAT* input4 = input[4];
    FAUSTFLOAT* input5 = input[5];
    FAUSTFLOAT* input6 = input[6];
    FAUSTFLOAT* input7 = input[7];
    FAUSTFLOAT* output0 = output[0];
    FAUSTFLOAT* output1 = output[1];
    FAUSTFLOAT* output2 = output[2];
    FAUSTFLOAT* output3 = output[3];
    FAUSTFLOAT* output4 = output[4];
    FAUSTFLOAT* output5 = output[5];
    FAUSTFLOAT* output6 = output[6];
    FAUSTFLOAT* output7 = output[7];

    /*	for (i=0;i<1024;i++)
        {
        input0[i]=input[0][i];
        input1[i]=input[1][i];
        input2[i]=input[2][i];
        input3[i]=input[3][i];
        input4[i]=input[4][i];
        input5[i]=input[5][i];
        input6[i]=input[6][i];
        input7[i]=input[7][i];
        }		
        */
    for (i=1; i<=count; i++) {
        fRec1[i] = (fSlow1 + (0.999f * fRec1[i-1]));
        fRec2[i] = (fSlow4 + (0.999f * fRec2[i-1]));
        fRec3[i] = (fSlow5 + (0.999f * fRec3[i-1]));

        fRec0[i] = (0 - (((fRec3[i] * fRec0[i-2]) + (fRec2[0] * fRec0[i-1])) - ((float)input0[i] * fRec1[i])));
        output0[i] = (fRec0[i] - fRec0[i-1]);


        fRec4[i] = (0 - (((fRec3[i] * fRec4[i-2]) + (fRec2[i] * fRec4[i-1])) - ((float)input1[i] * fRec1[i])));
        output1[i] = (fRec4[i] - fRec4[i-1]);

        fRec5[i] = (0 - (((fRec3[i] * fRec5[i-2]) + (fRec2[i] * fRec5[i-1])) - ((float)input2[i] * fRec1[i])));
        output2[i] = (fRec5[i] - fRec5[i-1]);

        fRec6[i] = (0 - (((fRec3[i] * fRec6[i-2]) + (fRec2[i] * fRec6[i-1])) - ((float)input3[i] * fRec1[i])));
        output3[i] = (fRec6[i] - fRec6[i-1]);

        fRec7[i] = (0 - (((fRec3[i] * fRec7[i-2]) + (fRec2[i] * fRec7[i-1])) - ((float)input4[i] * fRec1[i])));
        output4[i] = (fRec7[i] - fRec7[i-1]);

        fRec8[0] = (0 - (((fRec3[i] * fRec8[i-2]) + (fRec2[i] * fRec8[i-1])) - ((float)input5[i] * fRec1[i])));
        output5[i] = (fRec8[i] - fRec8[i-1]);

        fRec9[i] = (0 - (((fRec3[i] * fRec9[i-2]) + (fRec2[i] * fRec9[i-1])) - ((float)input6[i] * fRec1[i])));
        output6[i] = (fRec9[i] - fRec9[i-1]);

        fRec10[i] = (0 - (((fRec3[i] * fRec10[i-2]) + (fRec2[i] * fRec10[i-1])) - ((float)input7[i] * fRec1[i])));
        output7[i] = (fRec10[i] - fRec10[i-1]);

        // post processing
        /*fRec10[2] = fRec10[1]; fRec10[1] = fRec10[0];
          fRec9[2] = fRec9[1]; fRec9[1] = fRec9[0];
          fRec8[2] = fRec8[1]; fRec8[1] = fRec8[0];
          fRec7[2] = fRec7[1]; fRec7[1] = fRec7[0];
          fRec6[2] = fRec6[1]; fRec6[1] = fRec6[0];
          fRec5[2] = fRec5[1]; fRec5[1] = fRec5[0];
          fRec4[2] = fRec4[1]; fRec4[1] = fRec4[0];
          fRec0[2] = fRec0[1]; fRec0[1] = fRec0[0];
          fRec3[1] = fRec3[0];
          fRec2[1] = fRec2[0];
          fRec1[1] = fRec1[0];*/
    }
}

int main()
{int i;
    int count=1024;
    float in[8][1024];
    float out[8][1024];
    for (i=0;i<1024;i++)
    {
        in[0][i]=0;
        in[1][i]=1;
        in[2][i]=2;
        in[3][i]=3;
        in[4][i]=4;
        in[5][i]=5;
        in[6][i]=6;
        in[7][i]=8;
    }

    cry4pips(count,in,out);
    printf("%d-%d-%d-%d",out[2][123],out[4][321],out[6][1023],out[0][0]);
    return 0;
}


