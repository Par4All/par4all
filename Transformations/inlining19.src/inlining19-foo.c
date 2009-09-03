static const int kernel[3] = {1,2,3};
void foo(void)
{
  int i;
  int s = 0;
  for (i=0; i<3; i++)
    s += kernel[i];
}
