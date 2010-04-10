
int block_scope()
{
  int x = 6;
  {
  lab1: x--;
  x++;
  }
  {
    x = -1;
    goto lab1;
  }
  return x;
}
