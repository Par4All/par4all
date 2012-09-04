
// *y = i, le but est d'eliminer l'instruction inutile i = 2
int use_def_elim01()
{
  int i,  *x, *y;

  i = 2;
  x = &i;
  y = x;
  *y = 1;
  
  return *y;
}
