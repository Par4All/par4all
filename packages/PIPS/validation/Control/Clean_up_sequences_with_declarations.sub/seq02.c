int main()
{
  int i;
  {
    int j=0;
    i = j;
  }
  int k;
  {
    int l=i;
    k = l;
  }
  return k;
}
