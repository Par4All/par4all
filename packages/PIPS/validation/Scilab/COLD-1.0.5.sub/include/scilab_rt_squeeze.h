/*---------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010-2011
 *
 */

 
extern void scilab_rt_squeeze_d3_d2(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02],
															int sout00, int sout01,double out0[sout00][sout01]);															
															
extern void scilab_rt_squeeze_i3_i2(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02],
															int sout00, int sout01, int out0[sout00][sout01]);

extern void scilab_rt_squeeze_s3_s2(int sin00, int sin01, int sin02, char* in0[sin00][sin01][sin02],
															int sout00, int sout01, char* out0[sout00][sout01]);

extern void scilab_rt_squeeze_z3_z2(int sin00, int sin01, int sin02, double complex in0[sin00][sin01][sin02],
															int sout00, int sout01, double complex  out0[sout00][sout01]);

extern void scilab_rt_squeeze_i3_i3(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02],
															int sout00, int sout01, int sout02, int out0[sout00][sout01][sout02]);

extern void scilab_rt_squeeze_d3_d3(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02],
															int sout00, int sout01, int sout02, double out0[sout00][sout01][sout02]);
															
extern void scilab_rt_squeeze_z3_z3(int sin00, int sin01, int sin02, double complex in0[sin00][sin01][sin02],
															int sout00, int sout01, int sout02, double complex out0[sout00][sout01][sout02]);

extern void scilab_rt_squeeze_s3_s3(int sin00, int sin01, int sin02, char* in0[sin00][sin01][sin02],
															int sout00, int sout01, int sout02, char* out0[sout00][sout01][sout02]);
