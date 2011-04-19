/* Make sure that effects in declarations are taken into account */

int main()
{
     int *p,*q;
     int i,j;
     p = &i;
     q = &j;
     p = q;
     *p = 1;
  return 0;
}
