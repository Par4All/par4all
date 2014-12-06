/* Bug with update analysis: check subscripts */

void main()
{
  int a, tab[10];

  tab[a++]+=tab[a-1];
}
