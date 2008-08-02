int main()
{
  int x = 6;
  {
    int x = 7;
  }
  return x;
}

int x = 1;

void foo(int x)
{
  int x,y;
  if (x>1)
    {
      int x;
    }
  else 
    {
      int x;
    }
  if (y>1)
    {
      int x;
    }
  else
    {
      int y;
    }
}
