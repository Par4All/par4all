// Test case added by Vivien Maisonneuve.
//
// This test case was designed to show that a transition system could
// not be transformed into an automaton, if only because automata are
// restricted to regular languages. The langage here is A^kB^k for k
// in N (hence the tow counters c1 and c2).
//
// PIPS wise, the whole code should be reduced to the final printf
// because the printed value is known statically.

/*#define USE_VARIANT*/

#include <stdlib.h>
#include <stdio.h>

void error() {
	exit(1);
}

int alea() {
	return rand() % 2;
}

void run() {
	int s = 0, i = 0;
	int c1 = 0, c2 = 0; // compteurs pour la correction
#ifndef USE_VARIANT
	while (alea()) {
		i++;
		c1++;
	}
	s = 1;
	while (i >= 1) {
		i--;
		c2++;
	}
#else
	if (alea()) {
		i++;
		c1++;
		while (alea()) {
			i++;
			c1++;
		}
		s = 1;
		while (i >= 2) {
			i--;
			c2++;
		}
		i--;
		c2++;
	}
	else {
		s = 1;
	}
#endif
	if (c1 != c2) error();
	printf("i = %d\n", i);
}

int main(void) {
	run();
	return 0;
}

