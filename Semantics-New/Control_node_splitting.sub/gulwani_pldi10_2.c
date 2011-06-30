// Sumit Gulwani, Florian Zuleger: The reachability-bound problem.
// PLDI 2010: 292-304
// example 2

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (i <= n0)

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
#define LOOP(j) {while (rand_b()) {j}}

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

#define S1 CONTROL(n > 0 && m > 0 && s == 1)
#define S2 CONTROL(n > 0 && m == 0 && s == 1)
#define S3 CONTROL(n == 0 && s == 1)
#define S4 CONTROL(n > 0 && m > 0 && s == 2)
#define S5 CONTROL(n > 0 && m == 0 && s == 2)
#define S6 CONTROL(n == 0 && s == 2)

#define G1 (n > 0 && m > 0 && s == 1)
#define G1a (G1 && n > 1 && m > 1)
#define G1b (G1 && n > 1 && m == 1)
#define G1c (G1 && n == 1)
#define A1 {n--; m--; i++; s = 2;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)

#define G2 (n > 0 && s == 2)
#define G2a (G2 && n > 1)
#define G2b (G2 && n == 1)
#define A2 {n--; m++; i++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define A3 {s = 1;}
#define C3 COMMAND(G3, A3)

#define INI {n0 = rand(); m = rand(); n = n0; ASSUME(n > 0 && m > 0); s = 1; i = 0;}

// transition system

void ts_singlestate(void) {
	int n0, n, m, s, i;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int n0, n, m, s, i;
	INI; CHECK;
	S1;
	OR(
		LOOP(
			OR(C1a; S4, C1b; S5; C2a; S4);
			LOOP(C2a; S4);
			C3; S1
		);
		OR(C1a, C1b; S5; C2a);
		S4; LOOP(C2a; S4); C2b; S6,
	OR(
		C1b; S5; OR(C3; S2; deadlock();, C2b; S6),
		
		C1c; S6
	));
	C3; S3
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

