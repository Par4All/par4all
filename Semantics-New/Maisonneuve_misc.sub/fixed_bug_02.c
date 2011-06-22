// Segfaults in deprecated versions of PIPS.

// $Id$

#include <stdlib.h>

void run(void) {
	int n = 0;
	while (1) {
		if (rand()) {}
	}
}

int main(void) {
	run();
	return 0;
}

