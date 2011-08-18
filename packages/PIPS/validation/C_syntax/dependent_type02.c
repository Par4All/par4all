/* This code is not C99 compatible. You do not have a "self" to refer
   to your current data structures. hence min1, min2, max1 and max2
   are undefined. */

typedef enum {TRUE, FALSE} bool;

typedef struct allocatable_2d {
  bool allocated;
  int min1;
  int max1;
  int min2;
  int max2;
  double (*array)[max1-min1+1][max2-min2+2];
} allocatable_2d;

main()
{
  allocatable_2d a;
  return 0;
}
