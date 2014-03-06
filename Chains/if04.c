// use-def chains with if/else
// if and else don't work on same variable
// MAY Write

int if04()
{
  int r, r1=0, r2=0;

  if (1)
    r1 = 10;
  else
    r2 = 50;
  
  r= r1+r2;
  return r;
}
