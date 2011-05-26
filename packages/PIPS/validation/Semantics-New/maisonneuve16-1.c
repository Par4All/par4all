// We consider a 3-transition system:
// 
// T1: (G1) x in [0, 4], y in [2, 8], y >= x, y <= 10 - x ?
//     (U1) x' = x + 2, y' = y + 2
// 
// T2: (G2) x in [2, 8], y in [4, 10] ?
//     (U2) x' = x - 2, y' = y - 2
//
// T3: (G3) x in [2, 8], y in [0, 6] ?
//     (U3) x' = x - 2, y' = y + 2
// 	
// Initially, x and y satisfy G3.
// The goal is to show the state (x, y) = (1, 9) cannot be reached, while this
// state is in the convex hull of G1 union G2 union G3.
// 		
// Here is a first attempt, using a single-state automaton. It fails to check
// the property, even with transformer lists.

// $Id$

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

void run(void) {

	int x, y;
	x = rand();
	y = rand();
	if (x >= 5 && x <= 8 && y >= 0 && y <= 3) {

		while (1) {
			check_not(BAD);
			if (flip()) {
				T1;
			}
			else if (flip()) {
				T2;
			}
			else {
				T3;
			}
		}

	}

}

int main(void) {
	run();
	return 0;
}

