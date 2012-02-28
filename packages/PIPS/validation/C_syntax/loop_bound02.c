/* Bug with bound simplification */

#define N 20

void loop_bound02()
{
  int i, j;
  for(i=0;i<N-1;i++)
    for(j=0; j<i-1; j++)
      ;
  return;
}
