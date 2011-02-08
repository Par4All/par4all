// We consider the transition system defined in maisonneuve18-1. Its reachable set
// S = {(x, y) | x >= 0, y >= 0, (x <= 2 \/ y <= 2)}
// is split into two convex sets:
// S_L = {(x, y) | (x, y) in S and 2y < x+2}
// S_U = {(x, y) | (x, y) in S and 2y >= x+2}
// 
// The system is encoded as a 3-state
// automaton, whose control states are S_L, S_U and S_N the complement of S.
// Initially, x = y = 0 so the initial control state is S_L. State S_N is not reachable.//
//
// With this encoding, PIPS can check states so that (x >= 3 /\ y >= 3) are not
// reachable.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define BAD (x >= 3 && y >= 3)

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

#define G1 (y <= 2)
#define U1 {x++;}
#define T1 {trans(G1, U1);}

#define G2 (x <= 2)
#define U2 {y++;}
#define T2 {trans(G2, U2);}

#define S_L {assert(G1 && 2 * y < x + 2);}
#define S_U {assert(G2 && 2 * y >= x + 2);}

#define T1_LL T1
#define T2_LL {trans(G2 && 2 * y < x, U2);}
#define T1_UU {trans(G1 && 2 * y >= x + 3, U1);}
#define T2_UU T2
#define T1_UL {trans(G1 && 2 * y < x + 3, U1);}
#define T2_LU {trans(G2 && 2 * y >= x, U2);}

void run(void) {
	
	int x, y;
	x = y = 0;
	
	check_not(BAD);
	while (1) {
		S_L;
		if (flip()) {
			T1_LL; S_L;
		}
		else if (flip()) {
			T2_LL; S_L;
		}
		else {
			T2_LU; S_U;
			while (flip()) {
				if (flip()) {
					T1_UU; S_U;
				}
				else {
					T2_UU; S_U;
				}
			}
			T1_UL; S_L;
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

