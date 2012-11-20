/* Excerpt from fpointer01: issue with a read of *col
 *
 * OK for the excerpt...
 */

void fpointer02(int * col,
		int   indent)
{
  *col = 0;
  if(*col < indent)
    {
      ;
    }
  while(*col < indent)
    {
      ;
    }
}
