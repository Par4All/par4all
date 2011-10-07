// David Monniaux, Laure Gonnord: Using Bounded Model Checking to Focus
// Fixpoint Iterations. SAS 2011: 369-385
// Listing 5, p. 15.

// $Id$

// Transformer lists are required to find the expected invariant (-1000 <= x_old <= 1000).

#include <stdbool.h>
#include <stdlib.h>

void run(void) {
	int x_old, x;
	x_old = 0;
	while (1) {
		x = rand() - rand();
		if (x < -1000) exit(0);
		if (x > 1000) exit(0);
		if (x >= x_old + 1) x = x_old + 1;
		if (x <= x_old - 1) x = x_old - 1;
		x_old = x;
	}
}

int main(void) {
	run();
	return 0;
}

