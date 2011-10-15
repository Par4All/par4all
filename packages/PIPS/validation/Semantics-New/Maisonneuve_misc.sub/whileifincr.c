// Transformer lists are required to find the expected loop invariant (i <= 42).

// $Id$

#include <stdlib.h>

void error(void) {
	exit(1);
}

void whileifincr(void) {
	int i = 0;
	while (rand()) {
		if (i < 42) i++;
	}
}

int main(void) {
	whileifincr();
	return 0;
}

