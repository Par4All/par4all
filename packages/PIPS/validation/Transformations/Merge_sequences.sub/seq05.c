// three levels of nested sequences + entity names re-used in inner sequences
int main()
{
  int i = 0;
  {
    int j = 1;
    {
      int i = 2;
      j = j + i;
    }
    i = i +j;
  }
  return i;
}
