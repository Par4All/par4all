void foo(void)
{
  float a[3];
  int i = 1;
  // various no write effect statements that should be removed
  0;
  i;
  a;
  a[i];
  i+3;
}

void bla(void)
{
  foo();
}
