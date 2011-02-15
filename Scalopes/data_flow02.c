#include <stdio.h>

#define NB_BUF 10
#define SIZE 100

int main()
{
  int t, i, a[SIZE], b[SIZE/2], c[SIZE/2], s = 0;
  for(t = 0; t< NB_BUF; t++)
    {
    scmp_task_0:
      for (i=0; i<SIZE; i++)
	a[i] = i*t;
    scmp_task_1:
      for (i=0; i<SIZE/2; i++)
	b[i] = a[i];
    scmp_task_2:
      for (i=0; i<SIZE/2; i++)
	c[i] = a[i+SIZE/2];
    scmp_task_3:
      for (i=0; i<SIZE/2; i++)
		printf("%d\n", b[i]);
	//s = s + a[i];
    scmp_task_4:
      for (i=0; i<SIZE/2; i++)
		printf("%d\n", c[i]);
	//s = s + a[i];
    }
  return (0);
}

