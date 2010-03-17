// Check controlizer

void block04()
{
  int i = 1;
  {
    int j = i++;
    i += j;
  }
  i--;
}
