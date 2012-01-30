/*----------------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010
 *
 */

extern int scilab_rt_mopen_s0s0_(char* filename, char* flags);

extern void scilab_rt_mclose_i0_(int fd);

extern int scilab_rt_meof_i0_(int fd);

extern void scilab_rt_mfprintf_i0s0_(int fd, char* s);

extern void scilab_rt_mfprintf_i0s0d0_(int fd, char* s, double x);

extern double scilab_rt_mfscanf_i0i0s0_(int iter, int fd, char* format);

