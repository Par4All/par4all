#include <stdio.h>

/* Make sure that static functions receive unique names. */

static void print_hello()
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
