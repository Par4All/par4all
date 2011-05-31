/* Parallelizarion with pointers */

void pointer11(int n, float *p, float *q)
{
  int i;
  p = q;

  for(i=0; i<n; i++)
    p[i] = q[i];
}
