/*----------------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010
 *
 */

//WRITE TO SCILAB
extern void scilab_rt_write_to_scilab_s0b0_(char* s, int x);

extern void scilab_rt_write_to_scilab_s0i0_(char* s, int x);

extern void scilab_rt_write_to_scilab_s0d0_(char* s, double x);

extern void scilab_rt_write_to_scilab_s0s0_(char* s, char* str);

extern void scilab_rt_write_to_scilab_s0z0_(char* s, double complex x);

extern void scilab_rt_write_to_scilab_s0b2_(char* s, int xsize, int ysize, int aMatrix[xsize][ysize]);

extern void scilab_rt_write_to_scilab_s0i2_(char* s, int xsize, int ysize, int aMatrix[xsize][ysize]);

extern void scilab_rt_write_to_scilab_s0d2_(char* s, int xsize, int ysize, double aMatrix[xsize][ysize]);

extern void scilab_rt_write_to_scilab_s0s2_(char* s, int xsize, int ysize, char* aMatrix[xsize][ysize]);

extern void scilab_rt_write_to_scilab_s0z2_(char* s, int xsize, int ysize, double complex aMatrix[xsize][ysize]);

//READ FROM SCILAB
extern void scilab_rt_read_from_scilab_s0_b0(char* s, int* x);

extern void scilab_rt_read_from_scilab_s0_i0(char* s, int* x);

extern void scilab_rt_read_from_scilab_s0_d0(char* s, double* x);

extern void scilab_rt_read_from_scilab_s0_s0(char* s, char** str);

extern void scilab_rt_read_from_scilab_s0_z0(char* s, double complex* x);

extern void scilab_rt_read_from_scilab_s0_b2(char* s, int nx, int ny, int aMatrix[nx][ny]);

extern void scilab_rt_read_from_scilab_s0_i2(char* s, int nx, int ny, int aMatrix[nx][ny]);

extern void scilab_rt_read_from_scilab_s0_d2(char* s, int nx, int ny, double aMatrix[nx][ny]);

extern void scilab_rt_read_from_scilab_s0_s2(char* s, int nx, int ny, char* aMatrix[nx][ny]);

extern void scilab_rt_read_from_scilab_s0_z2(char* s, int nx, int ny, double complex aMatrix[nx][ny]);

int scilab_rt_read_int_from_scilab_s0_(char* in0);
    
double scilab_rt_read_real_from_scilab_s0_(char* in0);

double complex scilab_rt_read_complex_from_scilab_s0_(char* in0);

char* scilab_rt_read_string_from_scilab_s0_(char* in0);

void scilab_rt_read_intM_from_scilab_s0i0i0_i2(char* in0, int in1, int in2, int sout00, int sout01, int out0[sout00][sout01]);

void scilab_rt_read_realM_from_scilab_s0i0i0_d2(char* in0, int in1, int in2, int sout00, int sout01, double out0[sout00][sout01]);

void scilab_rt_read_complexM_from_scilab_s0i0i0_z2(char* in0, int in1, int in2, int sout00, int sout01, double complex out0[sout00][sout01]);

void scilab_rt_read_stringM_from_scilab_s0i0i0_s2(char* in0, int in1, int in2, int sout00, int sout01, char* out0[sout00][sout01]);


#ifdef UNDEF
extern void scilab_rt_read_from_scilab_s0_b0(char* s, int* x);

extern void scilab_rt_read_from_scilab_s0_i0(char* s, int* x);

extern void scilab_rt_read_from_scilab_s0_d0(char* s, double* x);

extern void scilab_rt_read_from_scilab_s0_s0(char* s, char** str);

extern void scilab_rt_read_from_scilab_s0_z0(char* s, double complex* x);

extern void scilab_rt_read_from_scilab_s0_b2(char* s, int* nx, int* ny, int (**aMatrix)[*nx][*ny]);

extern void scilab_rt_read_from_scilab_s0_i2(char* s, int* nx, int* ny, int (**aMatrix)[*nx][*ny]);

extern void scilab_rt_read_from_scilab_s0_d2(char* s, int* nx, int* ny, double (**aMatrix)[*nx][*ny]);

extern void scilab_rt_read_from_scilab_s0_s2(char* s, int* nx, int* ny, char* (**aMatrix)[*nx][*ny]);

extern void scilab_rt_read_from_scilab_s0_z2(char* s, int* nx, int* ny, double complex (**aMatrix)[*nx][*ny]);
#endif

