int i;

int main()
{
  int j = 0;
  i = 5;
  {
    int i = 3;
    j = j + i;
  }

  j = j + i;

  return j;
}
