// Pointer initializations to a 1-D array

// Same as init05.c, but assigments are used instead of initializations

// Goal: make sure we are providing consistant results

int main()
{
  int a[10];
  int * p;
  int (*q)[10];
  int * r;

  p = a;
  q = &a;
  r = &a[0];

  return 0;
}
