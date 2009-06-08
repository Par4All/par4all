/* Check that external initializationa are properly taken into
   account. */

void static02()
{
  static int i = 0.;

  i++;
  printf("%d\n", i);
}

main()
{
  static02();
  static02();
  static02();
}
