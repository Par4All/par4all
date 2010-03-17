// Check prettyprint of internal blocks with and without executable
// instructions

void block01()
{
  int i = 1;
  {
    int j = i++;
  }
  i++;
  {
    int k = i++;

    k++;
  }
}
