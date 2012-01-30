#include <stdio.h>
void scilab_rt_linspace_d0d0i0_d2(double in0, double in1,int in2, int sout00, int sout01, double out0[sout00][sout01]){


	double spc = 0.0;
	int i;
	
	printf("%g%g%g",in0,in1,spc);

	for (i = 0 ; i < in2; ++i){

		out0[0][i] = 1.0;

	}


}

