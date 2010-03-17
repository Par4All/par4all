#define FOO "This is the foo.c file"

typedef struct foo {
  int foo_first;
  int foo_second;
} t_foo;

static int foo_a(int k)
{
  return(k+1);
}

foo()
{
  int i;
  int j;

  i = foo_a(2);
  j = bar_b(3);
}
