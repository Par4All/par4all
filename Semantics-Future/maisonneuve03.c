// This example makes PIPS segfault (at least revisions 18437 & 18499)

// $Id$

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void run(void) {
	int x;

	while (flip()) {
		if (flip()) {
			if (x < 0) {
				x++;
			}
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

