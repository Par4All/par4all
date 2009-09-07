int foo(int i)
{
  return i;
}

int bla(int i)
{
  if (i>27)
    i = foo(i - 13);
  else if (i<-2)
    i = foo(i + 3);
  else
    i = foo(i + 2);
  return i;
}

int if03(int i)
{
  if (i>10)
    i = bla(i);
  else if (i<3)
    i = bla(i-1);
  else if (i<5)
    i = bla(i-2);
  else if (i<7)
    i = bla(i-3);
  else 
    i = bla(i+1);
  return i;
}
