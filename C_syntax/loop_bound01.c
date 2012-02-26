/* Bug with bound simplification */

#define N 10

void loop_bound01()
{
  int i;
  for(i=0;i<N-1;i++)
    ;
  return;
}
