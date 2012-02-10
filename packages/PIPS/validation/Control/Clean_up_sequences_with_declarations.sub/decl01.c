// rename a variable which is used inside a declaration initialization part

int main()
{
  int a[10];
  int i = 0;
  int k = 0;

  {
    int i = 1;

    {
      int j = a[i];
      k = j;
    }
  }
  return k;
}
