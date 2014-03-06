// use-def chains with 2 if
// MAY Write

int if02a()
{
  int r;

  if (1)
    r = 1;
  if (0)
    r = 0;
  
  return r;
}
