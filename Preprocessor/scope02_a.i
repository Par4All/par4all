extern int i;

void foo();

main()
{
  // printf("%d\n",i);
  foo();
}

static int i = 2;

void foo()
{
  printf("%d\n",i);
}
