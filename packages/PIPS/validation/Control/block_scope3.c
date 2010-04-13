
int block_scope()
{
  int x = 6;

  x--;
  {
    int x;
    x = 2;
    {
      int x;
      x = 3;
      x--;
    }
    if (x) {
      int x;
      x = 5;
    }
    x++;
  }
  return x;
}
