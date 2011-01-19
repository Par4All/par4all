// This examples works with one while loop,
// but not with two nested while loops
// (even with transformer lists)

// $Id$

#include <stdlib.h>

#define NESTEDWHILE

void error() {
	exit(1);
}

void run() {
	// two variables : a is always 0, b is always 1
	int a = 0, b = 1;
#ifdef NESTEDWHILE
	while (1) {
#endif
		while (1) {
			if (b < 1) error();
			b = a + 1;
			a = 0;
		}
#ifdef NESTEDWHILE
	}
#endif
}

int main(void) {
	run();
	return 0;
}

