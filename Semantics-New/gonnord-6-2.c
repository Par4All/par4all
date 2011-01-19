// Inspired from:
// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 6.2
// The system has 2 variables x, y (initially x = y = 0).
// There are 2 possible transitions:
// - y <= 1 ? x++;
// - y >= x - 1 ? x++; y++;
// The set of reachable states (x, y) is not convex.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
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

#if USE_CHECK == 0
#define check(e)
#define check_not(e)
#else
#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#endif

#define freeze {while(1);}

#ifdef GOOD
#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}
#else
#ifdef BAD
#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}
#else
#define trans(g, c) {if (g) {c;} else freeze;}
#endif
#endif

// transition system

#define G1 (y <= 1)
#define G1a (G1 && y > x - 1)
#define G1b (G1 && y == x - 1)
#define U1 {x++;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (y >= x - 1)
#define G2a (G2 && y < 1)
#define G2b (G2 && y == 1)
#define U2 {x++; y++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(G1 && G2);}
#define S2 {assert(G1 && !G2);}
#define S3 {assert(!G1 && G2);}

void run(void) {
	
	int x, y;
	x = y = 0;

#if RESTRUCTURE == 0
	
	while (1) {
		if (flip()) {
			T1;
		}
		else {
			T2;
		}
	}
	
#else

	S1;
	while (flip()) {
		if (flip()) {
			T1a; S1;
		}
		else {
			T2a; S1;
		}
	}
	if (flip()) {
		T1b; S2;
		while (1) {
			T1; S2;
		}
	}
	else {
		T2b; S3;
		while (1) {
			T2; S3;
		}
	}
	
#endif
	
}

int main(void) {
	run();
	return 0;
}

