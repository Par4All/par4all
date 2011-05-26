/* offsetof example : This macro with functional form returns the
   offset value in bytes of a member in the structure type.*/
#include <stdio.h>
#include <stddef.h>

struct mystruct {
  char singlechar;
  char arraymember[10];
  char anotherchar;
};

typedef struct mystruct str;

int main ()
{
  printf ("offsetof(mystruct,singlechar) is %lu\n",offsetof(str,singlechar));
  printf ("offsetof(mystruct,arraymember) is %lu\n",offsetof(str,arraymember));
  printf ("offsetof(mystruct,anotherchar) is %lu\n",offsetof(str,anotherchar));

  return 0;
}
