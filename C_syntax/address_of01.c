/* check the print out of &a[i*k*k] for Benchmark/linpackd.c */

typedef double REAL;


/* Bug in unsplit: the code of void is requested */
/* Bug in pretyprinter or in the prettyprinter: the "extern" keyword is lost */

int foo(int i,int * q)
{
}

REAL bar(int i,REAL * r)
{
}

address_of01(int i, REAL * b)
{
  int a[100];
  int j, k;
  int *p;
  REAL * q;
  REAL x;
  //extern void foo(int *);

  p = &a[i+j*k];
  i = j + foo(i, &a[i+j*k]);
  q = &b[i+j*k];
  x = x + bar(i, &b[i+j*k]);
}
