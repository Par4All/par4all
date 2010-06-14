/*
                      PM2 HIGH-PERF/ISOMALLOC
           High Performance Parallel Multithreaded Machine
                           version 2.0

             Gabriel Antoniu, Luc Bouge, Christian Perez,
                Jean-Francois Mehaut, Raymond Namyst

            Laboratoire de l'Informatique du Parallelisme
                        UMR 8512 CNRS-INRIA
                 Ecole Normale Superieure de Lyon

                      External Contributors:
                 Yves Denneulin (LMC - Grenoble),
                 Benoit Planquelle (LIFL - Lille)

                    1998 All Rights Reserved


                             NOTICE

 Permission to use, copy, modify, and distribute this software and
 its documentation for any purpose and without fee is hereby granted
 provided that the above copyright notice appear in all copies and
 that both the copyright notice and this permission notice appear in
 supporting documentation.

 Neither the institutions (Ecole Normale Superieure de Lyon,
 Laboratoire de L'informatique du Parallelisme, Universite des
 Sciences et Technologies de Lille, Laboratoire d'Informatique
 Fondamentale de Lille), nor the Authors make any representations
 about the suitability of this software for any purpose. This
 software is provided ``as is'' without express or implied warranty.
*/

#ifndef TIMINGS_EST_DEF
#define TIMINGS_EST_DEF
#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>
#include <sys/time.h>

typedef union {
  unsigned long long tick;
  struct {
    unsigned long low, high;
  } sub;
  struct timeval timev;
} Tick;

#ifdef PENTIUM_TIMINGS
#define GET_TICK(t)           __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#else
#define GET_TICK(t)           gettimeofday(&(t).timev, NULL)
#endif

#ifdef PENTIUM_TIMINGS
#define SET_TICK(tdest, tsource) tdest.tick = tsource.tick;
#else
#define SET_TICK(tdest, tsource) \
  tdest.timev.tv_sec = tsource.timev.tv_sec; \
  tdest.timev.tv_usec = tsource.timev.tv_usec;
#endif

void timings_init(void);

double timings_tick2usec(long long t);

double timings_stat(Tick t);

double timings_event();

extern long long _timings_residual;

extern Tick _last_timings_event;

#ifdef __cplusplus
}
#endif

#endif
