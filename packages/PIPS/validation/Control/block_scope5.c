/* Jump into a block from a goto before the block */
void block_scope()
{
  int x = 6;
  goto lab1;
 lab2:
  x = 2;
  {
    int x = 7;
  lab1:
    x--;
  }
  x++;
  goto lab2;
}
