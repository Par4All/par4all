int main()
{
  int x = 6;
  int z = 0;
  {
    int x = 7;
    z = z + x;
  }
  return x;
}
