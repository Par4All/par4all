/* Bug for i++ + (j-j). Excerpt from partial_eval18.c */

void foo(int i)
{
}

int main()
{
  int i;
  int j;
  i++ + (j-j);
  j;
  return i;
}
