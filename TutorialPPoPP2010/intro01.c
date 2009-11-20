int temp;
int
main (void)
{
  int i,j,c,a[100];
  c = 2;
  /* a simple parallel loop */
  for (i = 0;i<100;i++)
    {
      a[i] = c*a[i]+(a[i]-1);
    }
}
