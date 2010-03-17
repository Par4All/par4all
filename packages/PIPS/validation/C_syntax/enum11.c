/* Check the prettyprint of enum declarations: reference to enum or
   definition of enum

   additional nesting
 */

void enum11()
{
  struct {
    union {
      enum g {gaga} f3;
    } u;
  } s;
}
