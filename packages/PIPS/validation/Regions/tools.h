#ifndef __PYPS_VAL_TOOLS_H
#define __PYPS_VAL_TOOLS_H

#include <complex.h>

extern void init_data_file(const char* file);
extern void close_data_file();
extern void print_array_float(const char* name, const float* arr, const unsigned int n);
extern void print_array_int(const char* name, const int* arr, const unsigned int n);
extern void print_array_double(const char* name, const float* arr, const unsigned int n);
extern void print_array_long(const char* name, const long* arr, const unsigned int n);
extern void print_array_cplx(const char* name, const float complex* arr, const unsigned int n);
extern int init_data_float(float* ptr, const unsigned int n);
extern int init_data_double(double* ptr, const unsigned int n);
extern int init_data_long(long* ptr, const unsigned int n);
extern int init_data_int(int* ptr, const unsigned int n);
extern int init_data_cplx(float complex* ptr, const unsigned int n);

extern void init_args(int argc, char **argv);


#endif
