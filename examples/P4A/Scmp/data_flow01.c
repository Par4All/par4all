#include <stdio.h>

#define NB_BUF 10
#define SIZE 100

int main()
{
  int t, i, a[SIZE];
  for(t = 0; t< NB_BUF; t++)
    {
      printf("iteration %d\n", t);
    scmp_task_0:
      for (i=0; i<SIZE; i++)
	a[i] = i*t;
    scmp_task_1:
      for (i=0; i<SIZE; i++)
		printf("%d\n", a[i]);
	//s = s + a[i];
    }
  printf("THE END\n");
  return (0);
}

