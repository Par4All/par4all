/* TRAC Ticket 692 */

int dowhile01()
{
  int i, a[10], b[10];

  do {
    for (i=0; i<10; i++)
      a[i] = i;
  } while(0);

  do {
    for (i=0; i<10; i++)
      b[i] = a[i];
  } while (0);

  return 0;
}
