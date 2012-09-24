/* Bug found in EffectsWithPointsTo/array03.c
 *
 * Excerpt
 */

#define N 5
#define M 3

float d[N][M];

int foo(float (*b)[M])
{
  float c;
  (*b)[3] = 2.0;
  c = (*b)[3];
  b[1][3] = 2.0;
  c = b[1][3];
  
  ((*b)[3])++;
  (*b)[3] += 5.0;
  (b[1][3])++;
  b[1][3] += 5.0;

  return (1);
}

int main() 
{
  float a[N][M], ret;
  
  ret = foo(a);
  
  return 1;
}
