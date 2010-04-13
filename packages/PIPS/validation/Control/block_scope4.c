void block_scope()
{
  int x = 6;
  {
    int x;
  lab1:
    x--;
  }
  x++;
  goto lab1;
}
