// check use of constant increment and loop bound by semantics analysis
// Same as for03.c, but the absence of j makes the bounds for i in the
// loop less precise because i is no longer a multiple of j and 128.

//#include <stdio.h>

int for04()
{
  int x[512];
  int i;
  int j;

  for(i = 0/*, j = 0*/; i<500;i +=128/*, j++*/)
    x[i] = 0;

  //printf("%d, %d\n", i, j);

 return 0;
}

main()\
{
  for04();
}
