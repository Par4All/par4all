// We consider the same system as maisonneuve16-1'. It is restructured by
// partitioning in only two states:
//
// S1: G1 && x <= 1
//
// S2: (G1 || G2 || G3) && x >= 2
//
// These states are convex and the resulting transitions are convex too. With
// this restructuration, PIPS is able to check the property.
//
// The resulting system is relatively simple, unlike to what happens if one
// tries to restructure using guards (system too big to be properly analyzed).

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define BAD (x == 1 && y == 9)

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

#define G1 (x >= 0 && x <= 4 && y >= 2 && y <= 8 && y >= x && y <= 10 - x)
#define U1 {x += 2; y += 2;}
#define T1 {trans(G1, U1);}

#define G2 (x >= 2 && x <= 8 && y >= 4 && y <= 10)
#define U2 {x -= 2; y -= 2;}
#define T2 {trans(G2, U2);}

#define G3 (x >= 2 && x <= 8 && y >= 0 && y <= 6)
#define U3 {x -= 2; y += 2;}
#define T3 {trans(G3, U3);}

#define G12_1 (G1 && x <= 1)
#define U12_1 U1
#define T12_1 {trans(G12_1, U12_1);}
#define G22_1 (G1 && x >= 2)
#define U22_1 U1
#define T22_1 {trans(G22_1, U22_1);}

#define G21_2 (G2 && x <= 3)
#define U21_2 U2
#define T21_2 {trans(G21_2, U21_2);}
#define G22_2 (G2 && x >= 4)
#define U22_2 U2
#define T22_2 {trans(G22_2, U22_2);}

#define G21_3 (G3 && x <= 3)
#define U21_3 U3
#define T21_3 {trans(G21_3, U21_3);}
#define G22_3 (G3 && x >= 4)
#define U22_3 U3
#define T22_3 {trans(G22_3, U22_3);}

#define S1 {assert(x >= 0 && x <= 1 && y >= 2 && y <= 8);}
#define S2 {assert(x >= 2 && x <= 8 && y >= 0 && y <= 10);}

void run(void) {
		
	int x, y;
	x = rand();
	y = rand();
	if (x >= 5 && x <= 8 && y >= 0 && y <= 3) {

		while (1) {
			S2;
			check_not(BAD);
			while (flip()) {
				if (flip()) {
					T22_1; S2;
				}
				else if (flip()) {
					T22_2; S2;
				}
				else {
					T22_3; S2;
				}
			}
			if (flip()) {
				T21_2; S1;
			}
			else {
				T21_3; S1;
			}
			T12_1; S2;
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}

