// The transition system defined in maisonneuve18-1 is encoded as a 4-state
// automaton, depending on transition guards:
// S_12 : G1 and G2 are satisfied.
// S_1n2 : G1 is satisfied, not G2.
// S_n12 : G2 is satisfied, not G1.
// S_n1n2 : neither G1 nor G2 is satisfied.
//
// Initially, x = y = 0 so the initial control state is S_12. The system stays
// a while in S_12, then goes either in S_1n2 or S_n12 and remains in this
// state. State S_n1n2 is not reachable. States S_12, S_1n2, S_n12 are convex.
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

#define S_12 {assert(x >= 0 && x <= 2 && y >= 0 && y <= 2);}
#define S_1n2 {assert(x >= 3 && y >= 0 && y <= 2);}
#define S_n12 {assert(x >= 0 && x <= 2 && y >= 3);}

#define T1_12_12 {trans(G1 && x <= 1, U1);}
#define T2_12_12 {trans(G2 && y <= 1, U2);}
#define T1_12_1n2 {trans(G1 && x == 2, U1);}
#define T2_12_n12 {trans(G2 && y == 2, U2);}
#define T1_1n2_1n2 T1
#define T2_n12_n12 T2

void run(void) {
	
	int x, y;
	x = y = 0;
	
	S_12;
	check_not(BAD);
	while (flip()) {
		if (flip()) {
			T1_12_12; S_12;
		}
		else {
			T2_12_12; S_12;
		}
	}
	if (flip()) {
		T1_12_1n2; S_1n2;
		while (1) {
			T1_1n2_1n2; S_1n2;
		}
	}
	else {
		T2_12_n12; S_n12;
		while (1) {
			T2_n12_n12; S_n12;
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

