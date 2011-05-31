
void suppress_dead_code02()
{
  int i=1, j;

  /* Not dead code because of control effect */
  while(i) {
    j;
  }
}
