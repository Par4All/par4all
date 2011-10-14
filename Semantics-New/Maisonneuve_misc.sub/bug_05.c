// Segfaults while computing preconditions with PIPS r20506,
// using transformer lists.

// $Id$

#include <stdlib.h>

void run(void) {
lbl:
	exit(1);
	goto lbl;
}	

int main(void) {
	run();
	return 0;
}

