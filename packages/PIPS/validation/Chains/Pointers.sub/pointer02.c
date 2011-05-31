/* Parallelizarion with pointers */

void pointer02(int n, float *p)
{
  int i;
  float * r = p-1;

  for(i=0; i<n; i++)
    p[i] = r[i];
}
