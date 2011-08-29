extern void bla(int *);
int kernel[3] = {1,1,1};
int foo(void)
{
  bla(kernel);
  return 0;
}

