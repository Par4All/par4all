/* Check that test conditions are simplified when possible, not only
   when they are true or false globally */

void suppress_dead_code05(int *x)
{
  int i = 1, j, k;

  if(i>0 && j>0)
    k = 2;
  else
    k = 3;

  if(j>0 && i>0)
    k = 2;
  else
    k = 3;

  if(i<0 || j>0)
    k = 2;
  else
    k = 3;

  if(j>0 || i<0)
    k = 2;
  else
    k = 3;

  return;
}
