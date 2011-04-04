#include <stdio.h>

void counter00(int n)
{
  int i;
  // simple while
  i = 0;
  while (i<n)
    i++;
  // while with sequence
  i = 0;
  while (i<n)
  {
    i++;
    fprintf(stdout, "%d\n", i);
  }
  // empty while loop
  i=0;
  while (i++<n);
  // do loop
  for (i=0; i<n; i++)
    fprintf(stdout, "%d\n", i);
  // empty do loop
  for (i=0; i<n; i++);
  // for loop
  for (i=0; i<n && n<1000; i++)
    fprintf(stdout, "%d\n", i);
  // empty for loop
  for (i=0; i<n && n<1000; i++);
  // do-while loop
  i = 0;
  do {
    i++;
    fprintf(stdout, "%d\n", i);
  } while (i<n);
  // simple test
  if (n%2==0)
    i++;
  else
    i--;
  // sequence test
  if (n%2==0)
  {
    i++;
    fprintf(stdout, "%d\n", i);
  }
  else
  {
    i--;
    fprintf(stdout, "%d\n", i);
  }
  // test with true branch only
  if (n%2==0)
    i++;
}
