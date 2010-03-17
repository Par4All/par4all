#include <stdio.h>

/* What happens with a duplicate function name? */

void print_hello_too()
{
  printf("Hello World!\n");
}

void print_hello_too()
{
  printf("Hello World!\n");
}

main()
{
  print_hello();
  /* Let's print hello again! */
  print_hello_too();
}
