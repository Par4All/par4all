// Check prettyprint of internal blocks with and without executable
// instructions

void block01()
{
  int i = 1;
  {
    int j = 2;
  }
  i++;
  {
    int k = 2;

    k++;
  }
}
