/*---------------------------------------------------- -*- C -*-
*
*   (c) HPC Project - 2010-2011
*
*/

void scilab_rt_schur_i2_d2d2(int sin00, int sin01, int in0[sin00][sin01],
    int sout00, int sout01, double out0[sout00][sout01],
    int sout10, int sout11, double out1[sout10][sout11]);

void scilab_rt_schur_d2_d2d2(int sin00, int sin01, double in0[sin00][sin01],
    int sout00, int sout01, double out0[sout00][sout01],
    int sout10, int sout11, double out1[sout10][sout11]);

void scilab_rt_schur_z2_z2z2(int sin00, int sin01, double complex in0[sin00][sin01],
    int sout00, int sout01, double complex out0[sout00][sout01],
    int sout10, int sout11, double complex out1[sout10][sout11]);

