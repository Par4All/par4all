/* The loop is always entered and exited */

void foo(int i)
{
  ;
}

void complete01()
{
  int i;

  /* Redundant block to capture the completed loop transformer */
  {
    int j = 0;
    for(i = 0; i < 10; i++) {
      j++;
      ;
    }
  }

  foo(i);
}
