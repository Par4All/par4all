// This system has one variable n, whose value flips between 0 and 2
// Here PIPS is not able to find that n != 1, even with transformer lists

// $Id$

#include <stdlib.h>
#include <stdio.h>

int flip(void) {
	return rand() % 2;
}

void run(void) {

	int n = 0;
	while (1) {
		if (n == 1) printf("unreachable");
		if (n == 0) n = 2;
		else if (n == 2) n = 0;
	}

}

int main(void) {
	run();
	return 0;
}

