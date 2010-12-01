// This example is derived from bakery.c
// It is properly analyzed by PIPS r18437

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

	while (1) {

		a = rand();

		if (alea()) {
			// here, cannot see that b == 0
			if (alea() && b != 0) error();
			b = 0;
		}

	}
}

int main(void) {
  run();
  return 0;
}

