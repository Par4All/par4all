// this program reveals a problem with preconditions: PIPS claims that the
// `printf' instruction is unreachable, while it is actually reachable
// (tested on PIPS r18499, r18960)

// $Id$

#include <stdlib.h>
#include <stdio.h>

void run(void) {
	int n;
	n = 0;
	
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

