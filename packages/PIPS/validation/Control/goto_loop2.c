/* Test the continuation information of a goto loop */
int main() {
  int i = 5;

  /* The a label */
 a:
  i++;
  /* The b label */
 b:
  i--;
  /* Oh, go to a... */
  goto a;

  /* Unreachable... */
  return i;
}
