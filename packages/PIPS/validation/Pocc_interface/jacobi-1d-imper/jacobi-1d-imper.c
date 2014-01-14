#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

//#include "instrument.h"
#include <sys/time.h>

#ifndef NBTHREADS
# define  NBTHREADS 2
#endif
/* *********************************************** Pluto functions   */
/*int ceild(int n,int d)  
{
 	return ceil(((double)(n))/((double)(d)));
}
int floord(int n,int d) {return floor(((double)(n))/((double)(d)));}
int max(int x,int y)    {if ((x) > (y)) return x ;else  return y;}
int min(int x,int y)    {if ((x) < (y)) return x; else  return y;}*/
/******************************************************************* */
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}
double t_start, t_end;
/* Default problem size. */
#ifndef TSTEPS
# define TSTEPS 10000
#endif
#ifndef N
# define N 4096
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[N];
DATA_TYPE B[N];
#else
DATA_TYPE* A = (DATA_TYPE**)malloc(N * sizeof(DATA_TYPE));
DATA_TYPE* B = (DATA_TYPE**)malloc(N * sizeof(DATA_TYPE));
#endif


static inline
void init_array()
{
  int i;

  for (i = 0; i < N; i++)
    {
      A[i] = ((DATA_TYPE) 4 * i + 10) / N;
      B[i] = ((DATA_TYPE) 7 * i + 11) / N;
    }
}

/* Define the live-out variables. Code is not executed unless
   POLYBENCH_DUMP_ARRAYS is defined. */
static inline
void print_array(int argc, char** argv)
{
  int i;
#ifndef POLYBENCH_DUMP_ARRAYS
  if (argc > 42 && ! strcmp(argv[0], ""))
#endif
    {
      for (i = 0; i < N; i++) {
	fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
	if (i % 80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int t, i, j;
  int tsteps = TSTEPS;
  int n = N;

  /* Initialize array. */
  init_array();

  for (t = 0; t < tsteps; t++)
    {
      for (i = 2; i < n - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);

      for (j = 2; j < n - 1; j++)
	A[j] = B[j];
    }

  print_array(argc, argv);
  return 0;
}
