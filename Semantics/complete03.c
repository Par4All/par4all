/* The loop is always entered and never exited */

void foo(int i)
{
  ;
}

void complete03()
{
  int i;

  /* Redundant block to capture the completed loop transformer */
  {
    int j = 0;
    for(i = 0; i > 10; i++) {
      j++;
      i--;
    }
  }

  foo(i);
}
