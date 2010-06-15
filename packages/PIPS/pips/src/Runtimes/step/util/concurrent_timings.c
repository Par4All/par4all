#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include "trace.h"
#ifdef MPI
#include "mpi.h"
#endif
#ifdef THREAD
#include <pthread.h>
#endif

#ifdef TIMINGS
#include "timings.h"
#endif 
#ifdef IPM_TIMER
#include "IPM_timer.h"
IPM_timer ipm_t;
#endif

static int timings_nb_procs = 0;
static int timings_table_size = 0;
#if (defined TIMINGS || defined IPM_TIMER)
static double *timings_table = NULL;
#else
static struct timeval *timings_table = NULL;
#endif

void concurrent_timings_init(int nbprocs, int table_size)
{
  int i;

  IN_TRACE("nbprocs = %d, table_size = %d", nbprocs, table_size);

  timings_nb_procs = nbprocs;
  timings_table_size = table_size;

#ifdef DEBUG
  printf("timings_nb_procs = %d, timings_table_size = %d\n", timings_nb_procs, timings_table_size);
#endif

#if (defined TIMINGS || defined IPM_TIMER)
  timings_table = malloc(timings_nb_procs * timings_table_size * sizeof(double));
#else
  assert(sizeof(struct timeval) == 2 * sizeof(long));
  timings_table = malloc(timings_nb_procs * timings_table_size * sizeof(struct timeval));
#endif

  if (!timings_table)
    {
      perror("malloc");
      exit(EXIT_FAILURE);
    }
  for (i = 0 ;  i < timings_nb_procs * timings_table_size; i++)
#if (defined TIMINGS || defined IPM_TIMER)
    timings_table[i] = 0.0;
#else
    timings_table[i].tv_sec = 0L;
    timings_table[i].tv_usec = 0L;
#endif

#ifdef TIMINGS
  timings_init();
#endif
  OUT_TRACE("");
}

void concurrent_timings_finalize()
{
  IN_TRACE("");

  free(timings_table);
  
  timings_table = NULL; 
  OUT_TRACE("");
}

void concurrent_timings_start(int myrank, int timer)
{
#ifndef TIMINGS
#ifndef IPM_TIMER
  /* DEFAULT */
  struct timeval tv;
  int idx;
#endif
#endif

  IN_TRACE("myrank = %d, timer = %d", myrank, timer);


#ifdef TIMINGS
  timings_event();
#else
#ifdef IPM_TIMER
  IPM_timer_clear(&ipm_t);
  IPM_timer_start(&ipm_t);
#else

  /* DEFAULT */
  assert(timer < timings_table_size);
  idx = myrank * timings_table_size + timer;

  gettimeofday(&tv, NULL);

  timings_table[idx].tv_sec = tv.tv_sec;
  timings_table[idx].tv_usec = tv.tv_usec;

#endif
#endif

  OUT_TRACE("");
}

void concurrent_timings_stop(int myrank, int timer)
{
#ifndef TIMINGS
#ifndef IPM_TIMER
  /* DEFAULT */
  struct timeval tv;
#endif
#endif
  int idx;

  IN_TRACE("myrank = %d, timer = %d", myrank, timer);

  assert(timer < timings_table_size);

  idx = myrank * timings_table_size + timer;

#ifdef TIMINGS
  timings_table[idx] += timings_event();
  printf("%d/%d) timings_table[%d] = %g\n", myrank, timings_table_size, idx, timings_table[idx]);
#else
#ifdef IPM_TIMER
  IPM_timer_stop(&ipm_t);
  timings_table[idx] = IPM_timer_read(&ipm_t); 
#else 
  /* DEFAULT */
  gettimeofday(&tv, NULL);

  if (tv.tv_sec - timings_table[idx].tv_sec > (long int)(LONG_MAX / 1000000))
    {
      fprintf(stderr, "Oops tv_sec > (long int)(LONG_MAX) / 1000000\n");
      exit(EXIT_FAILURE);
    }


  timings_table[idx].tv_usec = (tv.tv_sec - timings_table[idx].tv_sec)*1000000 +  tv.tv_usec - timings_table[idx].tv_usec;
  timings_table[idx].tv_sec = -1L;

#endif
#endif

  OUT_TRACE("");
}

void concurrent_timings_print(char *filename)
{
  int p;
  FILE * outFile;

  IN_TRACE("filename = %p", filename);

  if (filename == NULL)
    outFile = stdout;
  else
    {
      outFile = fopen(filename, "a");
      if (!outFile)
	{
	  perror(filename);
	  exit(EXIT_FAILURE);
	}
    }

  fprintf(outFile, "Valeurs des differents timers :\n");
  for(p = 0; p < timings_nb_procs; p++) 
    {
      int i;
#ifdef MPI
      fprintf(outFile, "PROC %d ", p);
#else 
      fprintf(outFile, "THREAD %d ", p);
#endif
      for(i = 0; i < timings_table_size; i ++) 
	{
	  int idx;
	  idx = p * timings_table_size + i;
#if (defined TIMINGS || defined IPM_TIMER)
	  fprintf(outFile, "%f ", 
		  timings_table[idx]);
#else
	  fprintf(outFile, "TIMER %d USEC %ld ", 
		  i, timings_table[idx].tv_usec);
#endif
	}
      fprintf( outFile, "\n");
    }

  if (filename != NULL)
    fclose(outFile);

  OUT_TRACE("");
}

#ifdef MPI
void concurrent_timings_gather(int myrank, int rootrank)
{
  int idx;

  IN_TRACE("myrank = %d, rootrank = %d", myrank, rootrank);

  idx = myrank * timings_table_size;
#ifdef DEBUG
  printf("%d/%d) timings_table = %p, timings_table_size = %d \n", myrank, timings_nb_procs, timings_table, timings_table_size);
#endif

#if (defined TIMINGS || defined IPM_TIMER)
  MPI_Gather( &timings_table[idx], 
	      timings_table_size, MPI_DOUBLE,
	      timings_table, timings_table_size, MPI_DOUBLE, 
	      rootrank, MPI_COMM_WORLD);
#else
  /* struct timeval a transferer */

  MPI_Gather( &timings_table[idx], 
	      timings_table_size * 2, MPI_LONG,
	      timings_table, timings_table_size * 2, MPI_LONG, 
	      rootrank, MPI_COMM_WORLD);
#endif

  OUT_TRACE("");
}
#endif


#ifdef TEST

#ifdef THREAD
#define NBTHREADS 10
#endif

#define NBLOOPS 100
#define TABLE_SIZE 2
#define TIMER1 0
#define TIMER2 1

void *calcul(void *arg)
{
  int i;
  int *rank;
  float result;

  rank = arg; 

  concurrent_timings_start(*rank, TIMER1);

  result = 0.0;
  for (i = 0; i < NBLOOPS; i ++)
    {
      result += i * 2.3;
    }

  printf("%d) result = %f\n", *rank, result);

  concurrent_timings_stop(*rank, TIMER1);

  concurrent_timings_start(*rank, TIMER2);

  result = 0.0;
  for (i = 0; i < NBLOOPS * 1000; i ++)
    {
      result += i * 5.6;
    }

  printf("%d) result = %f\n", *rank, result);

  concurrent_timings_stop(*rank, TIMER2);

#ifdef THREAD
  free(rank);

  pthread_exit(NULL);
#else

  return NULL;
#endif
}

#ifdef MPI
int main(int argc, char **argv)
{
  int rank;
  int size;


  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /*  set_traces("traces", NULL, size, rank); */

  concurrent_timings_init(size, TABLE_SIZE);

  calcul(&rank);

  
  concurrent_timings_gather(rank, 0);

  if (rank == 0)
    concurrent_timings_print(NULL);

  #ifdef FRED
  concurrent_timings_finalize();
  printf("free plante\n");
  #endif

  MPI_Finalize();
  return EXIT_SUCCESS;
}
#endif

#ifdef THREAD
int main(int argc, char **argv)
{
  pthread_t threads[NBTHREADS];
  int i;
  int size;


  size = NBTHREADS;
  /*  set_traces("traces", NULL, size, rank); */

  concurrent_timings_init(size, TABLE_SIZE);

  for (i = 0; i < NBTHREADS; i ++)
    {
      int *rank;
      
      /* pour eviter: cast to pointer from integer of different size */
       rank = malloc(sizeof(int));
       if (!rank)
	 {
	   perror("malloc");
	   exit(EXIT_FAILURE);
	 }
       *rank = i;
       pthread_create(&threads[i], NULL, calcul, rank);
     }
   for (i = 0; i < NBTHREADS; i ++)
     {
       pthread_join(threads[i], NULL);
     }

  concurrent_timings_print(NULL);

#ifdef FRED
  concurrent_timings_finalize();
  printf("free plante\n");
#endif

  return EXIT_SUCCESS;
}
#endif
#endif
