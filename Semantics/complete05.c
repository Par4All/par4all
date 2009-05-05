/* The loop may be entered but never exited */

void foo(int i)
{
  ;
}

void complete05()
{
  int i;
  int n;

  /* Redundant block to capture the completed loop transformer */
  {
    int j = 0;
    for(i = 0; i < n; i++) {
      j++;
      i--;
    }
  }

  foo(i);
}
