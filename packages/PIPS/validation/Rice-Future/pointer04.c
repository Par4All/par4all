/* Parallelizarion with pointers
 *
 * Same as pointer02, but with a positive offset: the loop is not
 * parallel but it can be vectorized
 */

void pointer04(int n, float *p)
{
  int i;
  float * r = p+1;

  for(i=0; i<n; i++)
    p[i] = r[i];
}
