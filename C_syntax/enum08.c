/* Check the prettyprint of enum declarations: reference to enum or
   definition of enum

   Extension of enum06 and enum07 with nested declarations
 */

enum e
{
  a
};

void enum08()
{
  enum e i;
  enum f {x, y} j;
  struct {
    enum e f1;
    enum f f2;
    enum g {gaga} f3;
  } ss;
  i = a;
  j = x;
}
