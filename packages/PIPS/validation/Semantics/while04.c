/* Just like while01 and while02 and while03, but although the value
   of n is unknown, the loop is always entered and the increments are
   the same for i and j. A redundant while condition is used to see if
   it can be partially evaluated and the redundancy eliminated...

   This test is designed to see if while loops can be converted into
   for loops using a loop counter and preconditions on it. This is
   very unlikely because preconditions are not propagated into
   subexpressions so j<n cannot be evaluated using the loop
   precondition unioned with the body postcondition and anded with
   i<n-1.
 */

int while04(int n)
{
  int i, j;

  i = 0;
  j = 1;

  if(n<=1) exit(1);

  {int k;

  while(i<n-1 && j<n) {
    i++;
    j++;
  }
  }
  return i+j;
}
