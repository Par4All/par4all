// two internal sequence entities have the same same
int main()
{
  int i=0;
  {
    int j=1;
    i = i +j;
  }
  {
    int j= 2;
    i = i +j;
  }
  return i;
}
