// This system has one variable n, whose value flips between 0 and 2
// PIPS is not able to find that n != 1 in maisonneuve14-1.c, but here
// the k variable is added to generate an invariant

// $Id$

#include <stdlib.h>
#include <stdio.h>

int flip(void) {
	return rand() % 2;
}

void run(void) {

	int n = 0;
	int k = 1;
	while (1) {
		if (n == 1) printf("unreachable");
		if (n == 0) n = 2, k=0;
		else if (n == 2) n = 0, k=1;
	}

}

int main(void) {
	run();
	return 0;
}

