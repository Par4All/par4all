/* The loop may be entered and exited */

void foo(int i)
{
  ;
}

void complete04()
{
  int i;
  int n;

  /* Redundant block to capture the completed loop transformer */
  {
    int j = 0;
    for(i = 0; i < n; i++) {
      j++;
    }
  }

  foo(i);
}
