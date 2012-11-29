/* See how the lattice interact with targets of array elements...
 */

void array16()
{
  int a, b, c, d, e, f, g, h, i, j;
  int * p[10];
  int x[10];
  int ii;
  p[0]=&a;
  p[1]=&b;
  p[2]=&c;
  p[3]=&d;
  p[4]=&e;
  p[5]=&f;
  p[6]=&g;
  p[7]=&h;
  p[8]=&i;
  p[9]=&j;
  for(ii=0;ii<10;ii++)
    p[ii]=&x[ii];
  return;
}
