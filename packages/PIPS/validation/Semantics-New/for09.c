/* Bug with update analysis: although tab is an array, its assignments
   must be explored for side effects in the index expressions. */

int main()
{
  int a,b;
  int tab[10];

  b = 10;
  a = 1;
  for (int i=0; i<b; tab[a++]+=tab[a-1])
    i = i+1;
  return a+tab[a];
}
