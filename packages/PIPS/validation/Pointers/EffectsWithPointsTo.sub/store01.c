/* Check interface with effects_with_points_to */

int store01(int *p, int i)
{
  *p = i;
  return;
}
