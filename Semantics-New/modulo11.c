/* Exploit equations to evaluate modulo operations 
 *
 * Derived from a Dilig paper at OOPSLA 2013
*/

void modulo11(int flag)
{
   // PIPS: flag is assumed a constant reaching value
   int i, j = 1, a = 0, b = 0;
   float x;
   i = 1;
   while (x>0.) {
      a++;
      b += j-i;
      i += 2;
      if (i%2==0)
         j += 2;
      else
         j++;
   }
}
