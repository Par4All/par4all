/* effect less simplified statement: elimination */

void foo(int i)
{
}

int main()
{
  int i;
  int j;
  j;
  /* Can be eliminated because the store is not impacted */
  j+0;
  return i;
}
