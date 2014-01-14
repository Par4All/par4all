/* exemple pour le calcul de compatibilite */

// FI: I cannot believe that gcc does not warn about the uselesness of
// r and s in foo...

void foo(int ** pi, int **pj)
{
  int * r, *s;
  r = *pi;
  s = *pj;
  ** pi = 1, ** pj = 2;
  return;
}

int main()
{
  int **qq, **pp, *q, *p, i= 0, j = 1;
  p = &i, q = &j;
  pp = &p, qq = & q;
  foo(pp, qq);
  return 0;
}
