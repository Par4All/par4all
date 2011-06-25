// This example illustrates that a while-while structure may lead to less
// precise results than a single while structure.
// If run with PIPS r19837, the error function appears to be reachable in
// whilewhile but not in whilealone.

// $Id$

#include <stdlib.h>

void error(void) {
	exit(1);
}

void whilealone(void) {
	int a = 0, b = 1;
	while (1) {
	  if(1) {
			if (b < 1) error();
			b = a + 1;
			a = 0;
		}
	}
}

void whilewhile(void) {
	int a = 0, b = 1;
	while (1) {
		while (1) {
		  if(1) {
				if (b < 1) error();
				b = a + 1;
				a = 0;
		  }
		}
	}
}

int main(void) {
	if (rand()) whilealone();
	else whilewhile();
	return 0;
}

