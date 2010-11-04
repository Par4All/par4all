#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/time.h>

#define N 2048

double alpha = 1.54645764567;
double beta = 1.768679678;

double A[N][N];
double B[N][N];
double C[N][N];




#ifdef TIME
#define IF_TIME(foo) foo;
#else
#define IF_TIME(foo)
#endif

void init_array()
{
    int i, j;

    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
	  A[i][j] = C[i][j];
           
        }
    }
}



void print_array(char** argv)
{
    int i, j;
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
	if (N-i>2)
	   fprintf(stderr, "\n");
      }
    }
}


double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

#ifdef TIME
#define IF_TIME(foo) foo;
#else
#define IF_TIME(foo)
#endif





int main(int argc, char** argv)
{
    double t_start, t_end;
    int i, j, k;

    init_array();

    IF_TIME(t_start = rtclock());
    for (i=0; i<N; i++)
      for (j=0; j<N; j++) {
	C[i][j] = C[i][j] * alpha;
	for (k = 0; k < N; ++k)
	  C[i][j] += beta * A[i][k] * B[k][j];
      }


  for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
	  A[i][j] = C[i][j];
           
        }
    }

    IF_TIME(t_end = rtclock());
    IF_TIME(printf("%0.6lfs\n", t_end - t_start));


    print_array(argv);


    return 0;
}
