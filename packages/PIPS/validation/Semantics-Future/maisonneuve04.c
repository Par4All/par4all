// This example is derived from bakery.c
// It is properly analyzed by PIPS r18437

// This version is copied from maisonneuve01.c, but the test is
// replaced by while loops; the assignment of a was sinked into the
// test branches

#include <stdlib.h>

void error() {
	exit(1);
}

int alea() {
	return rand() % 2;
}

void run() {
	// two variables: a is never known, b is always 0
	int a;
	int b = 0;
	int c = alea();

	while (1) {


		while(c) {
			// here, cannot see that b == 0
			if (alea() && b != 0) error();
			a = rand();
			b = 0;
			c = alea();
		}

		while(!c) {
			// here, cannot see that b == 0
			if (alea() && b != 0) error();
			a = rand();
			c = alea();
		}

	}
}

int main(void) {
  run();
  return 0;
}

