/* Check the prettyprint of enum declarations: reference to enum or
   definition of enum

   Like enum08.c, but with union instead of struct
 */

enum e
{
  a
};

void enum09()
{
  enum e i;
  enum f {x, y} j;
  union {
    enum e f1;
    enum f f2;
    enum g {gaga} f3;
  } ss;
  i = a;
  j = x;
}
