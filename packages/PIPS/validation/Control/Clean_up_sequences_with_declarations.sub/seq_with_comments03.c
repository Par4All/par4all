// three levels of nested sequences + entity names re-used in inner sequences
// + comments
int main()
{
  int i = 0;
  {
    // comment on first statement of level 2 sequence
    int j = 1;
    {
      // comment on first statement of level 3 sequence
      int i = 2;
      j = j + i;
    }
    i = i +j;
  }
  return i;
}
