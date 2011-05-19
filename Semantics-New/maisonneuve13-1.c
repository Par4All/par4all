// This example requires transformer lists to be properly analyzed

// $Id$

#include <stdlib.h>
#include <stdio.h>

void run(void) {
	
	int x;
	x = 1 + rand();
	
	while (1) {
		if (x < 0) printf("unreachable\n");
		if (x > 0) x--;
		else if (x <= 0) x++;
	}
	
}

int main(void) {
	run();
	return 0;
}

