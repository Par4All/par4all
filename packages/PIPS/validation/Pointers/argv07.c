/* Representation of calling context for argv */

#include <stdio.h>

int argv07(int n, char * (*ap)[n]) 
{
  char *p = (*ap)[1];
  printf("\"%s\"\n", p);
  return p==p; // To silence gcc
}

int main(int argc, char ** argv)
{
  char * messages[10][10];
  messages[0][0] = "hello";
  messages[0][1] = "francois";
  messages[2][0] = "how";
  messages[3][0] = "are";
  messages[4][0] = "you";
  messages[5][0] = "doing";
  messages[6][0] = "tonight";
  (void) argv07(10, messages);
  return 0;
}
