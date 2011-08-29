int foo(int *x, int *y)
{
  int i;
  for(i=0 ; i<10 ; i++) {
// commenting next lines makes unfolding work...
    *x = 18;
  }
  return 0;
}

int main(int argc, char * argv[])
{
  int xshift, yshift;

  /* This program loops forever on purpose */
  while(1)
  {
    if (0) break;
    foo(&xshift,&yshift);
  }
  return 0;
}

