int main()
{
  int a,b;
  int tab[10];

  b = 10;
  for (a=0; a<b; tab[a]+=tab[a-1])
    a = a+1;
  return a+tab[a];
}
