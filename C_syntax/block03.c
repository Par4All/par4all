// Check controlizer

void block03()
{
  int i = 1;
  {
    int j = i++;
    int i;

    i += j;
  }
}
