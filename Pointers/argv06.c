/* Representation of calling context for argv */

/* This declaration generates a warning in gcc if the function name is
   "main". */

#include <stdio.h>

int argv06(int argc, char * (*argv)[argc]) 
{
  char *p = (void *) 0;
  argv++;
  p = (*argv)[2];
  printf("%s\n", p);
  return p==p; // To silence gcc
}

int main()
{
  char * a[10][10];
  int i, j, n=10;

  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      asprintf(&a[i][j], "%d", 10*i+j);

  argv06(10, a);
  return 0;
}
