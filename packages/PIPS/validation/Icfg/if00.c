int bla(int i)
{
  if (i>27)
    i = i - 13;
  else
    i = i + 2;
  return i;
}

int if00(int i)
{
  if (i>10)
    i = bla(i);
  else
    i = i+1;
  return i;
}
