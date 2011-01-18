// BAKERY MUTUAL EXCLUSION ALGORITHM
// Leslie Lamport: A New Solution of Dijkstra's Concurrent Programming Problem.
// Commun. ACM 17(8): 453-455 (1974)

// $Id$

#define USE_ASSERT 1
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

#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#define freeze {while(1);}

// transition system

#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}

#define T 1
#define W 2
#define C 3

#define G1 (s1 == T)
#define C1 {a = b + 1; s1 = W;}
#define T1 {trans(G1, C1);}

#define G2 (s1 == W && (a < b || b == 0))
#define C2 {s1 = C;}
#define T2 {trans(G2, C2);}

#define G3 (s1 == C)
#define C3 {a = 0; s1 = T;}
#define T3 {trans(G3, C3);}

#define G4 (s2 == T)
#define C4 {b = a + 1; s2 = W;}
#define T4 {trans(G4, C4);}

#define G5 (s2 == W && (b < a || a == 0))
#define C5 {s2 = C;}
#define T5 {trans(G5, C5);}

#define G6 (s2 == C)
#define C6 {a = 0; s2 = T;}
#define T6 {trans(G6, C6);}

void run(void) {
	
	int s1, s2, a, b;
	s1 = s2 = T;
	a = b = 0;

	while (1) {

		if (rand() % 2) {T1; T2; T3;} else {if (rand() % 2) {T4;} else {T1; if (rand() % 2) {T4; T2; T3;} else {T2; T4; T3;}} while (rand() % 2) {if (rand() % 2) {T1; T5;} else {T5; T1;} T6; if (rand() % 2) {T4; T2; T3;} else {T2; T4; T3;}} if (rand() % 2) {if (rand() % 2) {T1; T5;} else {T5; T1;} T6; T2; T3;} else {T5; T6;}}
		
	}
}

int main(void) {
	run();
	return 0;
}

