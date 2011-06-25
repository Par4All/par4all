// We consider the same system as maisonneuve15-1' and maisonneuve15-2'. This
// time, we restructure it by partitioning in only two states:
//
// S1: G1 && y <= 6 - x
//
// S2: G2 && y >= 7 - x
//
// Contrary to what happened in maisonneuve15-2, these states are convex and
// the resulting transitions are convex too. With this restructuration, PIPS is
// able to check the property, both with and without state assertions.
//
// Also notice the resulting system is simpler than in maisonneuve15-2, since
// it contains only 2 states instead of 3.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define BAD (x == 1 && y == 5)

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

#define G1 (x >= 0 && x <= 4 && y >= 0 && y <= 4)
#define G1a (G1 && y <= 2 - x)
#define G1b (G1 && y >= 3 - x)
#define U1 {x += 2; y += 2;}
#define T1 {trans(G1, U1);}
#define T1a {trans(G1a, U1);}
#define T1b {trans(G1b, U1);}

#define G2 (x >= 2 && x <= 6 && y >= 2 && y <= 6)
#define G2a (G2 && y >= 11 - x)
#define G2b (G2 && y <= 10 - x)
#define U2 {x -= 2; y -= 2;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}

#define S1 {assert(G1 && y <= 6 - x);}
#define S2 {assert(G2 && y >= 7 - x);}

void run(void) {
	
	int x, y;
	x = rand();
	y = rand();

	if (G1 && y <= 6 - x) {
		
		while (1) {
			S1;
			while (flip()) {
				if (flip()) {
					T1a; S1;
				}
				else {
					T2; S1;
				}
			}
			T1b; S2;
			while (flip()) {
				if (flip()) {
					T1; S2;
				}
				else {
					T2a; S2;
				}
			}
			T2b; S1;
		}
		
	}
	else if (G2 && y >= 7 - x) {
		
		while (1) {
			S2;
			while (flip()) {
				if (flip()) {
					T1; S2;
				}
				else {
					T2a; S2;
				}
			}
			T2b; S1;
			while (flip()) {
				if (flip()) {
					T1a; S1;
				}
				else {
					T2; S1;
				}
			}
			T1b; S2;			
		}

	}
	
}

int main(void) {
	run();
	return 0;
}

