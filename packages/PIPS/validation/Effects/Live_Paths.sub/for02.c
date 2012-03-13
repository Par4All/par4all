int main()
{
  int a,b;
  int tab[10];

  b = 10;
  for (a=0; a<b; tab[a]+=tab[a-1]) {
    int k;
    k = a;
    a = a+1+k;
  }
  return a+tab[a];
}
