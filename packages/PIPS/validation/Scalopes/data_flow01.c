#include <stdio.h>

#define NB_BUF 10
#define SIZE 100

int main()
{
  int t, i, a[SIZE], s = 0;
  for(t = 0; t< NB_BUF; t++)
    {
    scmp_task_0:
      for (i=0; i<SIZE; i++)
	a[i] = i*t;
    scmp_task_1:
      for (i=0; i<SIZE; i++)
		printf("%d\n", a[i]);
	//s = s + a[i];
    }
  return (0);
}

