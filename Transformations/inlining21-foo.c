extern void bla(int *);
static const int kernel[3] = {1,1,1};
int foo(void)
{
  bla(kernel);
  return 0;
}

