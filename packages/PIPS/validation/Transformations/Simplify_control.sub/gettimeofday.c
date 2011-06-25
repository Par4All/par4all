#include <sys/time.h>



double get_time(void)
{
   struct timeval t;
   struct timezone tzp;
   gettimeofday(&t, &tzp);
   return t.tv_sec+t.tv_usec*1e-6;
}

