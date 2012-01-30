/* (c) HPC Project 2010 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int scilab_rt_sum_i0_(int in0);

void scilab_rt_sum_i0_i0(int in0, int* out0);

int scilab_rt_sum_i0s0_(int in0, char* in1);

void scilab_rt_sum_i0s0_i0(int in0, char* in1, int* out0);

double scilab_rt_sum_d0_(double in0);

void scilab_rt_sum_d0_d0(double in0, double* out0);

double scilab_rt_sum_d0s0_(double in0, char* in1);

void scilab_rt_sum_d0s0_d0(double in0, char* in1, double* out0);

void scilab_rt_sum_i2_i0(int sin00, int sin01, int in0[sin00][sin01], int* out0);

void scilab_rt_sum_d2_d0(int sin00, int sin01, double in0[sin00][sin01], double* out0);

void scilab_rt_sum_i3_i0(int sin00, int sin01, int sin02, int in0[sin00][sin01][sin02], int* out0);

void scilab_rt_sum_d3_d0(int sin00, int sin01, int sin02, double in0[sin00][sin01][sin02], double* out0);

void scilab_rt_sum_i2s0_i0(int sin00, int sin01, int in0[sin00][sin01], char* in1, int *out0);

void scilab_rt_sum_d2s0_d0(int sin00, int sin01, double in0[sin00][sin01], char* in1, double *out0);

void scilab_rt_sum_i2s0_i2(int sin00, int sin01, int in0[sin00][sin01], char* in1,
    int sout00, int sout01, int out0[sout00][sout01]);

void scilab_rt_sum_d2s0_d2(int sin00, int sin01, double in0[sin00][sin01], char* in1,
    int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_sum_i2i0_i0(int sin00, int sin01, int in0[sin00][sin01], int in1, int *out0);

void scilab_rt_sum_d2i0_d0(int sin00, int sin01, double in0[sin00][sin01], int in1, double *out0);

void scilab_rt_sum_i2i0_i2(int sin00, int sin01, int in0[sin00][sin01], int in1,
    int sout00, int sout01, int out0[sout00][sout01]);

void scilab_rt_sum_d2i0_d2(int sin00, int sin01, double in0[sin00][sin01], int in1,
    int sout00, int sout01, double out0[sout00][sout01]);

