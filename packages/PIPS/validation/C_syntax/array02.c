/* check the impact of sizeof() for Serge */

array02()
{
  int a[sizeof(int)];
  int b[4];
  int i;

  for(i=0; i<sizeof(int);i+=sizeof(char))
    a[i] = 0.;
}
