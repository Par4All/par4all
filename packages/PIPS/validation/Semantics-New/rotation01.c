/* Vivien suggested a non affine bounded domain
 *
 * Here is a case with 8 points in 2-D.
 *
 * The case should be trivial for ASPIC because the number of states
 * is bounded and small.
 *
 * The case is much harder for a transformer-based approach, because
 * the transformations are not as easy to combine as the states.
 *
 * Transformer lists are necessary.
 */

void rotation01()
{
  int x = 1, y = 0;
  while(1) {
    if(x==1&&y==0)
      x++;
    if(x==2&&y==0)
      x++, y++;
    if(x==3&&y==1)
      y++;
    if(x==3&&y==2)
      x--, y++;
    if(x==2&&y==3)
      x--;
    if(x==1&&y==3)
      x--,y--;
    if(x==0&&y==2)
      y--;
    if(x==0&&y==1)
      x++,y--;
  }
}
