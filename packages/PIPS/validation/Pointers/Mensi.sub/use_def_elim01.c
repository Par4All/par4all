
// *y = i, le but est d'eliminer l'instruction inutile i = 2
int use_def_elim01()
{
  int i, j,  *x, *y;

  i = 2;
  x = &j;
  x = &i;
  y = x;
  *y = 1;
  
  return *y;
}
