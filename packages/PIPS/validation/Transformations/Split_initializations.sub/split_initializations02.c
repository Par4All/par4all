/* Expected result: these declarations cannot be split, because their
   initial values allocate memory dynamically.

   NOTE: this test case is identical to flatten_code_07.c
 */

void split_initializations02(void)
{
  int k[] = { 1, 2, 3 };
  {
    int k[] = { 1, 2, 3 };
  }
  if (1)
  {
    int k[] = { 1, 2, 3 };
  }
}
