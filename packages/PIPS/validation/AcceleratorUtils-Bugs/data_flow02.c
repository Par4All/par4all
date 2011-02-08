#include <stdio.h>

#define NB_BUF 10
#define SIZE 100
void task_1(int a[SIZE], int t)
{
  int i;
      for (i=0; i<SIZE; i++)
	a[i] = i*t;
}

void task_2(int a[SIZE])
{
  int i, s = 0;
  for (i=0; i<SIZE; i++)
    printf("%d\n", a[i]);
  //s = s+a[i];
}
int main()
{
  int t, i, a[SIZE], s = 0;
  for(t = 0; t< NB_BUF; t++)
    {
    scmp_task_0:
      task_1(a,t);
    scmp_task_1:
      task_2(a);
    }
  return (0);
}

