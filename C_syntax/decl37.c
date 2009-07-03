/* Check that the variable "b" is not considered wrongly declared twice.
 *
 * Problem shown in C_syntax/tpips.tpips within function parse_arguments.
 *
 * The initialization of variable "a" to NULL causes the bug.
*/

#define NULL ((void *) 0)

typedef char * string;

void decl37()
{
  //int i = 0, j = 1;
  //string a = NULL, b = NULL;
  string a = NULL, b;
}
