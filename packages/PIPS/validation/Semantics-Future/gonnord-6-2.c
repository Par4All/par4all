// NON-CONVEX REACHABLE SET
// Inspired from:
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires; figure 6.2
// The system has 2 variables x, y (initially x = y = 0).
// There are 2 possible transitions:
// - y <= 1 ? x++;
// - y >= x - 1 ? x++; y++;
// The set of reachable states (x, y) is not convex.

// $Id$

#define USE_ASSERT 0
#define BAD (x == 4 && y == 2)

// usefull stuff

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void assert_error(void) {
	exit(1);
}

void checking_error(void) {
	exit(2);
}

#if USE_ASSERT == 0
#define assert(e)
#else
#define assert(e) {if (!(e)) assert_error();}
#endif

#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

#define G1 (y <= 1)
#define G2 (y >= x - 1)

#define C1 {x++;}
#define C2 {x++; y++;}

void run(void) {
	int x, y;
	x = y = 0;

	assert(G1 && G2);
	check_not(BAD);
	while (flip()) {
		while (flip()) {
			trans(G1 && y > x - 1, C1);
		}
		while (flip()) {
			trans(G2 && y < 1, C2);
		}
		assert(G1 && G2);
	}
	if (flip()) {
		trans(G1 && y == x - 1, C1);
		assert(G1 && !G2);
		while (1) {
			trans(G1, C1);
			assert(G1 && !G2);
		}
	}
	else {
		trans(G2 && y == 1, C2);
		assert(!G1 && G2);
		while (1) {
			trans(G2, C2);
			assert(!G1 && G2);
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

