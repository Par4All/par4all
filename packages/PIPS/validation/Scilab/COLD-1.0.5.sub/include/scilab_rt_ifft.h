/*---------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010-2011
 *
 */

void scilab_rt_ifft_i2_z2(int sin00, int sin01, int in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_ifft_d2_z2(int sin00, int sin01, double in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_ifft_z2_z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01]);

