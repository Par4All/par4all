
#include <stdlib.h>

#define check(c) if (!(c)) error()

void error() {
	exit(1);
}

// this examples works with one while loop,
// but not with two nested while loops

#define WW

void run() {
	// two variables : a is always 0, b is always 1
	int a = 0, b = 1;
#ifdef WW
	while (1) {
#endif
		while (1) {
			check(b >= 1);
			b = a + 1;
			a = 0;
		}
#ifdef WW
	}
#endif
}

int main(void) {
	run();
	return 0;
}

