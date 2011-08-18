/* _Bool is a gcc predefined type*/

/* ISO C Standard:  7.16  Boolean type and values  <stdbool.h> */

#include <stdbool.h>

int bool02()
{
  _Bool i;

  i = true;
  return (int) i;
}
