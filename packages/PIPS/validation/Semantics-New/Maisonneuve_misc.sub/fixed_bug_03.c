// Bad reachability analysis in deprecated versions of PIPS, when using
// transformer lists.

// $Id$

#include <stdlib.h>
#include <stdio.h>

void run(void) {
	int n = 0;
	while (1) {
		if (rand() % 2) {
			if (n <= 9) n++;
		}
		else {
			if (n == 10) {
				printf("reachable instruction!\n");
				n = 0;
			}
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

