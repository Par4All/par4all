struct thing {
  int stuff;
};

void foo(struct thing * f)
{
  f->stuff++;
}

void bla(struct thing * b)
{
  foo(b);
}

int main(void)
{
  struct thing t;
  t.stuff = 1;
  bla(&t);

  struct thing * s;
  s = &t;
  bla(s);
  return 0;
}
