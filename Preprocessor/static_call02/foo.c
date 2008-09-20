static void foo(int i)
{
  int foo;
  void * p = (void *) &foo;
  /* This is the static version in foo.c */
  i++;

  printf("This is the static version in foo.c\n");
}

void bar(int i)
{
  foo(i);

  if(i) {
    extern void foo(int i);

    foo(i);
  }
}
