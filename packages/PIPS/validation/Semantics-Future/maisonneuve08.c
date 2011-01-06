// this program combine three different uses of b: identity, reset and
// increment, but in the loop b is always positive

// $Id: $

#include <stdlib.h>
#include <stdio.h>

int alea() {
  return rand() % 2;
}

void run(void)
{
int b = 0;
while(1) {
	if(alea())
		;
	else if(alea())
		b = 0;
	else
		b++;
}}

int main(void)
{
  run();
  return 0;
}

