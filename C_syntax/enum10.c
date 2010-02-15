/* Check the prettyprint of enum declarations: reference to enum or
   definition of enum

   Simplified version of enum09.c, for easier debugging
 */

void enum10()
{
  union {
    enum g {gaga} f3;
  } ss;
}
