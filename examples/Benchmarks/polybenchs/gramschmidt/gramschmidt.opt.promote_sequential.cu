/* Use the Par4All accelerator run time: */
#include <p4a_accel.h>
/*
 * file for gramschmidt.c
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "../../common/timing.h"

/* Default problem size. */







/* Default data type is double. */







double nrm;
double A[2048][2048];
double R[2048][2048];
double Q[2048][2048];

static void init_array();


/* Define the live-out variables. Code is not executed unless
 POLYBENCH_DUMP_ARRAYS is defined. */
static void print_array(int argc, char **argv);


int main(int argc, char **argv);
//PIPS generated variable
void p4a_launcher_main(double R[2048][2048], int n), p4a_launcher_main_1(double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m), p4a_launcher_main_2(double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n), p4a_launcher_main_3(double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n), P4A_runtime_copy_to_accel(void *host_ptr, size_t size), P4A_runtime_copy_from_accel(void *host_ptr, size_t size);
//PIPS generated variable
void *P4A_runtime_host_ptr_to_accel_ptr(void *host_ptr, size_t size);
//PIPS generated variable
P4A_accel_kernel_wrapper p4a_wrapper_main(int j, int k, double R[2048][2048], int n), p4a_wrapper_main_1(int i, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m), p4a_wrapper_main_2(int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n), p4a_wrapper_main_3(int i, int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n);
//PIPS generated variable
P4A_accel_kernel p4a_kernel_main(int j, int k, double R[2048][2048], int n), p4a_kernel_main_1(int i, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m), p4a_kernel_main_2(int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n), p4a_kernel_main_3(int i, int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n);
P4A_accel_kernel_wrapper p4a_wrapper_main_3(int i, int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n)
{
   // Index has been replaced by P4A_vp_1:
   j = P4A_vp_1;
   // Index has been replaced by P4A_vp_0:
   i = P4A_vp_0;
   // Loop nest P4A end
   p4a_kernel_main_3(i, j, A, Q, R, k, m, n);
}
P4A_accel_kernel p4a_kernel_main_3(int i, int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n)
{
   if ((j<=(((-k)+n)-2))&&(i<=(m-1)))
      A[i][(j+k)+1] = (A[i][(j+k)+1]-(Q[i][k]*R[k][(j+k)+1]));
}
P4A_accel_kernel_wrapper p4a_wrapper_main_2(int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n)
{
   // Index has been replaced by P4A_vp_0:
   j = P4A_vp_0;
   // Loop nest P4A end
   p4a_kernel_main_2(j, A, Q, R, k, m, n);
}
P4A_accel_kernel p4a_kernel_main_2(int j, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n)
{
   //PIPS generated variable
   int i;
   if (j<=(((-k)+n)-2))
      for(i = 0; i <= m-1; i += 1)
         R[k][(j+k)+1] += (Q[i][k]*A[i][(j+k)+1]);
}
P4A_accel_kernel_wrapper p4a_wrapper_main_1(int i, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m)
{
   // Index has been replaced by P4A_vp_0:
   i = P4A_vp_0;
   // Loop nest P4A end
   p4a_kernel_main_1(i, A, Q, R, k, m);
}
P4A_accel_kernel p4a_kernel_main_1(int i, double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m)
{
   if (i<=(m-1))
      Q[i][k] = (A[i][k]/R[k][k]);
}
P4A_accel_kernel_wrapper p4a_wrapper_main(int j, int k, double R[2048][2048], int n)
{
   // Index has been replaced by P4A_vp_1:
   k = P4A_vp_1;
   // Index has been replaced by P4A_vp_0:
   j = P4A_vp_0;
   // Loop nest P4A end
   p4a_kernel_main(j, k, R, n);
}
P4A_accel_kernel p4a_kernel_main(int j, int k, double R[2048][2048], int n)
{
   if ((k<=(n-1))&&(j<=(((-k)+n)-2)))
      R[k][(j+k)+1] = 0;
}
void p4a_launcher_main_3(double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n)
{
   //PIPS generated variable
   int i, j;
   P4A_call_accel_kernel_2d(p4a_wrapper_main_3, m, ((-k)+n)-1, ((int) i), ((int) j), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(A, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(Q, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(R, (sizeof(double)*4194304))), ((int) k), ((int) m), ((int) n));
}
void p4a_launcher_main_2(double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m, int n)
{
   //PIPS generated variable
   int j;
   P4A_call_accel_kernel_1d(p4a_wrapper_main_2, ((-k)+n)-1, ((int) j), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(A, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(Q, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(R, (sizeof(double)*4194304))), ((int) k), ((int) m), ((int) n));
}
void p4a_launcher_main_1(double A[2048][2048], double Q[2048][2048], double R[2048][2048], int k, int m)
{
   //PIPS generated variable
   int i;
   P4A_call_accel_kernel_1d(p4a_wrapper_main_1, m, ((int) i), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(A, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(Q, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(R, (sizeof(double)*4194304))), ((int) k), ((int) m));
}
void p4a_launcher_main(double R[2048][2048], int n)
{
   //PIPS generated variable
   int j, k;
   P4A_call_accel_kernel_2d(p4a_wrapper_main, (((-2+n)<-1)?-1:(-2+n))+1, n, ((int) j), ((int) k), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(R, (sizeof(double)*4194304))), ((int) n));
}
static void init_array()
{
   int i, j;

   i = 0;
   while (i<2048) {
      j = 0;
      while (j<2048) {
         A[i][j] = ((((double) (i+1))*(j+1))/2048);
         j++;
      }
      i++;
   }
}
static void print_array(int argc, char **argv)
{
   int i, j;

   if ((argc>42)&&(!strcmp(argv[0], ""))) {
      for(i = 0; i <= 2047; i += 1)
         for(j = 0; j <= 2047; j += 1) {
            fprintf(stderr, "%0.2lf ", A[i][j]);
            if ((((i*2048)+j)%80)==20)
               fprintf(stderr, "\n");
         }
      fprintf(stderr, "\n");
   }
}

__global__ void sequential_kernel(double A[2048][2048], double R[2048][2048], int m, int k) {
  int nrm,i;
  nrm = 0;
  for (i = 0; i <= m - 1; i += 1)
    nrm += (A[i][k] * A[i][k]);
  R[k][k] = sqrtf(nrm);
}

int main(int argc, char **argv)
{
   P4A_init_accel;
   int i, k;
   int m = 2048;
   int n = 2048;
   
   /* Initialize array. */
   init_array();
   
   /* Start timer. */
   timer_start();
   
   /* Cheat the compiler to limit the scope of optimisation */
   if (argv[0]==0)
      init_array();

   p4a_launcher_main(R, n);
   P4A_runtime_copy_to_accel(A, (sizeof(double)*4194304));
   P4A_runtime_copy_to_accel(R, (sizeof(double)*4194304));
   for(k = 0; k <= n-1; k += 1) {
      sequential_kernel<<<1,1>>>(((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(A, (sizeof(double)*4194304))), ((double (*)[2048]) P4A_runtime_host_ptr_to_accel_ptr(R, (sizeof(double)*4194304))), m,k);
      p4a_launcher_main_1(A, Q, R, k, m);
      p4a_launcher_main_2(A, Q, R, k, m, n);
      p4a_launcher_main_3(A, Q, R, k, m, n);
   }
   
   /* Cheat the compiler to limit the scope of optimisation */
   if (argv[0]==0)
      print_array(argc, argv);
   
   /* Stop and print timer. */
   timer_stop_display();

   print_array(argc, argv);

   return 0;
}
