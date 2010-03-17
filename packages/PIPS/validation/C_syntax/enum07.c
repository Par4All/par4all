/* Check the prettyprint of enum declarations: reference to enum or
   definition of enum

   Extension of enum06
 */

enum e
{
  a
};

void enum07()
{
  enum e i;
  enum f {x, y} j;
  enum f k;
  i = a;
  j = x;
}
