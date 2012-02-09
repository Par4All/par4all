#include <stdio.h>



void STAP_PulseComp(int tv, float ptrin[tv], int tf, float ptrfiltre[tf],
		float ptrout[tv-tf+1]) {

	int i, j,k;
	float R, I;
		for (k=0; k<32; k++) {
	
	for (i=0; i<tv-tf+1; i++) {

		R = 0.0;
		I = 0.0;
		for (j=0; j<tf; j++) {
			R += ptrin[i+j] * ptrfiltre[j] - ptrin[i+j]
					* ptrfiltre[j];
			I += ptrin[i+j] * ptrfiltre[j] + ptrin[i+j]
					* ptrfiltre[j];
		}
		ptrout[i] = R;
		ptrout[i] = I;
	}
}
		// printf ("result %f, %f",ptrout[1],ptrout[2]   );
}

void trt (int tv, int tf,float in_pulse[5][32][95],float out_pulse[5][32][80])
{
	//int tv=95; int tf=16; 
	int lv=20;
	int j,k,i;
	float filtre[16];
		
	//init_array();
	for (i=0; i<90; i++) {
		
	for (j=0; j<5; j++) {
		for (k=0; k<32; k++) {
	
			STAP_PulseComp(tv, in_pulse[j][k], tf, filtre, out_pulse[j][k]);
		}
	}
	}
}
int main ()
{
  int tv=95;
  int tf=16;
  float in_pulse[5][32][95];
  float out_pulse[5][32][80];

  trt(tv, tf, in_pulse, out_pulse);
  return 0;
}
