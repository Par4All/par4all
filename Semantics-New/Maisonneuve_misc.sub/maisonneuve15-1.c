// We consider a 2-transition system:
// 
// T1: (G1) x in [0, 4], y in [0, 4] ?
//     (U1) x' = x + 2, y' = y + 2
// 
// T2: (G2) x in [2, 6], y in [2, 6] ?
//     (U2) x' = x - 2, y' = y - 2
// 	
// Initially, x and y satisfy G1 or G2.
// The goal is to show the state (x, y) = (1, 5) cannot be reached, while this
// state is in the convex hull of G1 union G2.
// 		
// Here is a first attempt, using a single-state automaton. It fails to check
// the property, even with transformer lists.

// $Id$

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
#define U1 {x += 2; y += 2;}
#define T1 {trans(G1, U1);}

#define G2 (x >= 2 && x <= 6 && y >= 2 && y <= 6)
#define U2 {x -= 2; y -= 2;}
#define T2 {trans(G2, U2);}

void run(void) {
	
	int x, y;
	x = rand();
	y = rand();
	if (G1 || G2) {

		check_not(BAD);
		while (1) {
			if (G1 && flip()) {
				T1;
			}
			else if (G2 && flip()) {
				T2;
			}
		}
		
	}
	
}

int main(void) {
	run();
	return 0;
}

