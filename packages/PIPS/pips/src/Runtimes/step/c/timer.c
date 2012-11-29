#include "step_api.h"
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef INLINE
#define INLINE static inline
#endif
#include "array.h"

typedef struct
{
  uint64_t last_start;
  Array events;
} Timer;


// time in micro seconds
static void tick(uint64_t *usec)
{
  struct timeval tv;

  gettimeofday(&tv, NULL);
  *usec = (uint64_t)tv.tv_sec*1000000+(uint64_t)tv.tv_usec;
}


void STEP_API(step_timer_init)(size_t *timer_)
{
  Timer *timer = malloc(sizeof(*timer));

  *timer_ = (size_t)timer;
  array_set(&(timer->events), uint64_t);

  tick(&(timer->last_start));
}

void STEP_API(step_timer_event)(size_t *timer_)
{
  Timer *timer;
  uint64_t events[2];

  tick(&(events[1]));

  timer = (Timer*) *timer_;
  events[0] = timer->last_start;
  array_append_vals(&(timer->events), events, 2);

  tick(&(timer->last_start));
}

void STEP_API(step_timer_dump)(size_t *timer_, char *filename, STEP_ARG *id_file, STEP_ARG filename_len)
{
  Timer *timer;
  char filename_[filename_len+1];
  char *file_name;
  uint32_t len;
  FILE *fd;
  uint32_t id_event;

  step_timer_event_(timer_);

  timer = (Timer*) *timer_;

  if(*id_file > 0)
    len = (uint32_t)floorl(log10l(*id_file));
  else
    len=0;
  len += filename_len + 2;

  file_name = malloc(len);
  strncpy(filename_, filename, filename_len);
  filename_[filename_len]=0;
  snprintf(file_name, len, "%s%Lu", filename_, (unsigned long long)*id_file);

  fd = fopen(file_name,"a+");

  for(id_event=0; id_event<timer->events.len; id_event++)
    {
      uint64_t time = array_get_data_from_index(&(timer->events), uint64_t, id_event);
      fprintf(fd, "%Lu ", (unsigned long long)time);
    }
  fprintf(fd, "\n");

  fclose(fd);
  free(file_name);

  tick(&(timer->last_start));
}

void STEP_API(step_timer_finalize)(size_t *timer_)
{
  Timer *timer;

  timer = (Timer*) *timer_;
  array_unset(&(timer->events));
  free(timer);
  *timer_ = 0;
}

void STEP_API(step_timer_reset)(size_t *timer_)
{
  Timer *timer;

  timer = (Timer*) *timer_;
  array_reset(&(timer->events), NULL, 0);
  tick(&(timer->last_start));
}
