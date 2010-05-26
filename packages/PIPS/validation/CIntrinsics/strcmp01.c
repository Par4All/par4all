/* strcmp example : Compare two strings */
#include <stdio.h>
#include <string.h>

int main ()
{
  char szKey[] = "apple";
  char szInput[80];
  do {
     printf ("Guess my favourite fruit? ");
     gets (szInput);
  } while (strcmp (szKey,szInput) != 0);
  puts ("Correct answer!");
  return 0;
}
