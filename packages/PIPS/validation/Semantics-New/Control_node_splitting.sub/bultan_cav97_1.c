// Leslie Lamport: A New Solution of Dijkstra's Concurrent Programming Problem.
// Commun. ACM 17(8): 453-455 (1974)

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (s1 == C && s2 == C)

// tools

#include <stdlib.h>
#include <stdio.h>

int rand_b(void) {
	return rand() % 2;
}
int rand_z(void) {
	return rand() - rand();
}

#define OR(t1, t2) {if (rand_b()) {t1} else {t2}}
#define LOOP(t) {while (rand_b()) {t}}

void deadlock() {
	printf("deadlock\n");
	while (1);
}
#define ASSUME(c) {if (!(c)) deadlock();}

#if DO_CONTROL == 0
#define CONTROL(c)
#else
void control_error(void) {
	fprintf(stderr, "control error");
	exit(1);
}
#define CONTROL(c) {if (!(c)) control_error();}
#endif

#if DO_CHECKING == 0
#define CHECK
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#ifdef GOOD
#define CHECK {if (!(GOOD)) checking_error();}
#else
#ifdef BAD
#define CHECK {if (BAD) checking_error();}
#endif
#endif
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK;}

// control and commands

#define S1 CONTROL(s1 == T && s2 == T && a == 0 && b == 0)
#define S2 CONTROL(s1 == W && s2 == T && a > b && b == 0)
#define S3 CONTROL(s1 == W && s2 == W && b > a && a > 0)
#define S4 CONTROL(s1 == C && s2 == T && a > b && b == 0)
#define S5 CONTROL(s1 == C && s2 == W && b > a && a > 0)
#define S6 CONTROL(s1 == T && s2 == W && b > a && a == 0)
#define S7 CONTROL(s1 == W && s2 == W && a > b && b > 0)
#define S8 CONTROL(s1 == T && s2 == C && b > a && a == 0)
#define S9 CONTROL(s1 == W && s2 == C && a > b && b > 0)

#define T 1
#define W 2
#define C 3

#define G1 (s1 == T)
#define A1 {a = b + 1; s1 = W;}
#define C1 COMMAND(G1, A1)

#define G2 (s1 == W && (a < b || b == 0))
#define A2 {s1 = C;}
#define C2 COMMAND(G2, A2)

#define G3 (s1 == C)
#define A3 {a = 0; s1 = T;}
#define C3 COMMAND(G3, A3)

#define G4 (s2 == T)
#define A4 {b = a + 1; s2 = W;}
#define C4 COMMAND(G4, A4)

#define G5 (s2 == W && (b < a || a == 0))
#define A5 {s2 = C;}
#define C5 COMMAND(G5, A5)

#define G6 (s2 == C)
#define A6 {b = 0; s2 = T;}
#define C6 COMMAND(G6, A6)

#define INI {\
	s1 = s2 = T;\
	a = b = 0;\
}

// transition system

void ts_singlestate(void) {
	int s1, s2, a, b;
	INI; CHECK;
	LOOP(
		OR(C1,
		OR(C2,
		OR(C3,
		OR(C4,
		OR(C5,
		C6)))))
	)
}

void ts_restructured(void) {
	int s1, s2, a, b;
	INI; CHECK;
	LOOP(
		S1;
		OR(
			C1; S2; C2; S4; C3; S1,

			OR(
				C4; S6,
				C1; S2; OR(C4; S3; C2; S5; C3; S6, C2; S4; C4; S5; C3; S6)
			);
			LOOP(
				OR(
					C1; S7; C5; S9,
					C5; S8; C1; S9
				);
				C6; S2;
				OR(
					C4; S3; C2; S5; C3; S6,
					C2; S4; C4; S5; C3; S6
				)
			)
			OR(
				OR(
					C1; S7; C5; S9,
					C5; S8; C1; S9
				);
				C6; S2; C2; S4; C3; S1,

				C5; S8; C6; S1
			)
		)
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

