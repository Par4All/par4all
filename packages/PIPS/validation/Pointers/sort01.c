// Check the sorting function

int * foo()
{
  return (void *) 0;
}

void sort01()
{
  int a[10];
  int *p=foo(), *q=foo();
  if(p==q)
    p = &a[4];
  else
    p = &a[2];
  return;
}
