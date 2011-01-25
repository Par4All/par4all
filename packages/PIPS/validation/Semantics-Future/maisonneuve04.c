// This version is copied from maisonneuve01.c, but the test is
// replaced by while loops; the assignment of a was sinked into the
// test branches

// This example requires transformer lists to be properly analyzed

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define GOOD (1)

// usefull stuff

#include <stdlib.h>

void error() {
	exit(1);
}

void run() {
	// two variables: a is never known, b is always 0
	int a;
	int b = 0;
	int c = rand() % 2;

	while (1) {


		while(c) {
			if (rand() % 2 && b != 0) error();
			a = rand();
			b = 0;
			c = rand() % 2;
		}

		while(!c) {
			if (rand() % 2 && b != 0) error();
			a = rand();
			c = rand() % 2;
		}

	}
}

int main(void) {
  run();
  return 0;
}

