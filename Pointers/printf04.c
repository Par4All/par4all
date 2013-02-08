/* A constant string is the effective parameter of error() 
 *
 * Bug...
 */

#include <stdio.h>
#include <stdlib.h>


void error(const char * msg)
{
  printf("%s\n", msg);
}

int main(void)
{
      error("ici");

  return 0;
}
