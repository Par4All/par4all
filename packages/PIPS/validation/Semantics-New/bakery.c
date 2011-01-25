// BAKERY MUTUAL EXCLUSION ALGORITHM
// Leslie Lamport: A New Solution of Dijkstra's Concurrent Programming Problem.
// Commun. ACM 17(8): 453-455 (1974)

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define BAD (s1 == C && s2 == C)

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

#define T 1
#define W 2
#define C 3

#define G1 (s1 == T)
#define U1 {a = b + 1; s1 = W;}
#define T1 {trans(G1, U1);}

#define G2 (s1 == W && (a < b || b == 0))
#define U2 {s1 = C;}
#define T2 {trans(G2, U2);}

#define G3 (s1 == C)
#define U3 {a = 0; s1 = T;}
#define T3 {trans(G3, U3);}

#define G4 (s2 == T)
#define U4 {b = a + 1; s2 = W;}
#define T4 {trans(G4, U4);}

#define G5 (s2 == W && (b < a || a == 0))
#define U5 {s2 = C;}
#define T5 {trans(G5, U5);}

#define G6 (s2 == C)
#define U6 {b = 0; s2 = T;}
#define T6 {trans(G6, U6);}

#define S1 {assert(s1 == T && s2 == T && a == 0 && b == 0);}
#define S2 {assert(s1 == W && s2 == T && a > b && b == 0);}
#define S3 {assert(s1 == W && s2 == W && b > a && a > 0);}
#define S4 {assert(s1 == C && s2 == T && a > b && b == 0);}
#define S5 {assert(s1 == C && s2 == W && b > a && a > 0);}
#define S6 {assert(s1 == T && s2 == W && b > a && a == 0);}
#define S7 {assert(s1 == W && s2 == W && a > b && b > 0);}
#define S8 {assert(s1 == T && s2 == C && b > a && a == 0);}
#define S9 {assert(s1 == W && s2 == C && a > b && b > 0);}

void run(void) {
	
	int s1, s2, a, b;
	s1 = s2 = T;
	a = b = 0;

	while (1) {

#if RESTRUCTURE == 0

		if (flip()) {
			T1;
		}
		else if (flip()) {
			T2;
		}
		else if (flip()) {
			T3;
		}
		else if (flip()) {
			T4;
		}
		else if (flip()) {
			T5;
		}
		else if (flip()) {
			T6;
		}
		
#else

		S1;
		if (flip()) {
			T1; S2; T2; S4; T3; S1;
		}
		else {
			if (flip()) {
				T4; S6;
			}
			else {
				T1; S2;
				if (flip()) {
					T4; S3; T2; S5; T3; S6;
				}
				else {
					T2; S4; T4; S5; T3; S6;
				}
			}
			while (flip()) {
				if (flip()) {
					T1; S7; T5; S9;
				}
				else {
					T5; S8; T1; S9;
				}
				T6; S2;
				if (flip()) {
					T4; S3; T2; S5; T3; S6;
				}
				else {
					T2; S4; T4; S5; T3; S6;
				}
			}
			if (flip()) {
				if (flip()) {
					T1; S7; T5; S9;
				}
				else {
					T5; S8; T1; S9;
				}
				T6; S2; T2; S4; T3; S1;
			}
			else {
				T5; S8; T6; S1;
			}
		}
		
#endif
		
	}
}

int main(void) {
	run();
	return 0;
}

