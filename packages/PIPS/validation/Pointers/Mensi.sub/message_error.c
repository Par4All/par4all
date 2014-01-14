#include<stdio.h>
#include<stdlib.h>
void msg_error(char* msg)
{
  printf("%s", msg);
  exit(1);
}

int main()
{
  char *msg1 = "hello";
  char *msg2 = msg1;
  msg_error(msg2);

  return 0;
}
