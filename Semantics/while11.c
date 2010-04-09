/* To check a conjecture by Corinne that variables not used in the
   while condition did not receive their final values, here y == 51,
   but an inequality, here it would have been y >= 51.

   Well, PIPS find the correct postcondition for the while loop.
*/

void while11()
{
  int x = 0;
  int y = 0;
  while(x<=50) {
    x++;
    y++;
  }
  y = 2;
}
