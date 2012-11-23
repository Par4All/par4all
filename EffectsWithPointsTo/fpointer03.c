/* Excerpt of fpointer01.c. Bug for proper read effects of
 * while(*col<indent).
 */

void fpointer03(char c,
		int * col,
		int   indent,
		int * nbout)
{
  /* The reading of _col_2 was lost here... */
  while(*col < indent) {
    (*col)++;
  }

  (*col)++;
}
