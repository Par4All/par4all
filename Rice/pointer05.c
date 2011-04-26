/* Parallelizarion with pointers */

void pointer05(int n, float p[n], float q[n])
{
  int i;

  for(i=0; i<n; i++) {
    float x = q[i];
    p[i] = x;
  }
}
