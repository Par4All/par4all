// If run with PIPS r19837, this example requires transformer lists
// to be properly analyzed.

// $Id$

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void run(void) {
	int j;
	j = 0;
	
	while (flip()) {
		if (j > 8) {
			"unreachable";
		}
		if (flip()) {
			if (j < 8) {j++;}
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

