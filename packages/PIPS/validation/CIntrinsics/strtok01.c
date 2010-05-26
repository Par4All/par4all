/* Split string into tokens */
#include <stdio.h>
#include <string.h>

int main()
{

  static char str[] = "?a???b,,,#c";
  char *t;
  printf ("debut: str=%s",str);
  t = strtok(str, "?"); // t points to the token "a"
  printf ("\nresultat strtok(str,\"?\")= %s",t);
  t = strtok(NULL, ","); // t points to the token "??b"
  printf ("\nresultat strtok(NULL,\",\")= %s",t); 
  t = strtok(NULL, "#,"); // t points to the token "c"
  printf ("\nresultat strtok(NULL,\"#,\")= %s",t); 
  t = strtok(NULL, "?"); // t is a null pointer
  printf ("\nresultat strtok(NULL,\"?\")= %s",t);
}
