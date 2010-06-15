#ifndef _CONCURRENT_TIMINGS_H
#define _CONCURRENT_TIMINGS_H
#ifdef __cplusplus
extern "C" 
{
#endif

void concurrent_timings_init(int nbprocs, int table_size);
void concurrent_timings_finalize();
void concurrent_timings_start(int myrank, int timer);
void concurrent_timings_stop(int myrank, int timer);
#ifdef MPI
void concurrent_timings_gather(int myrank, int rootrank);
#endif
void concurrent_timings_print(char *filename);

#ifdef __cplusplus
}
#endif

#endif
