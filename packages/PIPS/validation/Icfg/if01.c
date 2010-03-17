int bla(int i)
{
  if (i>27)
    i = i - 13;
  else
    i = i + 2;
  return i;
}

int if01(int i)
{
  if (i>10)
    i = bla(i);
  else if (i<3)
    i = bla(i-1);
  else 
    i = bla(i+1);
  return i;
}
