/* TRAC Ticket 692 */

// Simplified version of dowhile01. I want to make sure that the
// transformer of the do while loop is correct... and it is not

// However, the postcondition i==10 is correct in the last
// transformer. It is not longer found in dowhile01 preconditions.

int dowhile02()
{
  int i, a[10], b[10];

  do {
    for (i=0; i<10; i++)
      a[i] = i;
  } while(0);

  return 0;
}
