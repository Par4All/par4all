/* Check the varag case */
void generate10()
{
  extern int func(int a, ...);
  int i, j, k;

  func(i, j, k);
}
