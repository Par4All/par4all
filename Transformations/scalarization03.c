int main(int argc, char **argv)
{
  int d, i, n=100;
  int x[n], y[n], t[n];
  int *x2, *y2, *t2;
  typedef int myarray[100]; // guess what ?
  myarray x3, y3, t3;

  /* TEST 1: scalarize an array.
     Expected results:
     a) t[i] should be scalarized
     b) a declaration should be created for the new scalar(s)
  */
  for (i=0 ; i<n ; i++) {
    t[i] = x[i];
    x[i] = y[i];
    y[i] = t[i];
  }

  /* TEST 2: scalarize a pointer
     Expected results:
     a) t2[i] should be scalarized 
     b) a declaration should be created for the new scalar(s)
  */
  for (i=0 ; i<n ; i++) {
    t2[i] = x2[i];
    x2[i] = y2[i];
    y2[i] = t2[i];
  }

  /* TEST 3: scalarize an array hidden behind a typedef
     Expected results:
     a) t3[i] should be scalarized 
     b) a declaration should be created for the new scalar(s)
  */
  for (i=0 ; i<n ; i++) {
    t3[i] = x3[i];
    x3[i] = y3[i];
    y3[i] = t3[i];
  }

  d = 1;
  return d;
}
