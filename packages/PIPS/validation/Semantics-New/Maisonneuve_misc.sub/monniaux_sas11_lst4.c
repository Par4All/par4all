// David Monniaux, Laure Gonnord: Using Bounded Model Checking to Focus
// Fixpoint Iterations. SAS 2011: 369-385
// Listing 4, p. 15.

// $Id$

// Transformer lists are required to find one of the expected invariants (-1 <= d <= 1).
// The other invariant (0 <= x <= 1000) is never found.

#include <stdbool.h>
#include <stdlib.h>

void run(void) {
	int x = 0;
	int d = 1;
	while (true) {
		if (x == 0) d = 1;
		if (x == 1000) d = -1;
		x += d;
	}
}

int main(void) {
	run();
	return 0;
}

