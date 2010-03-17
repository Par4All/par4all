struct foo {int a[4];};

struct foo f();

int struct02(int index)
{
  return f().a[index];
}
