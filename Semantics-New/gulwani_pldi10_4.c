// Sumit Gulwani, Florian Zuleger: The reachability-bound problem.
// PLDI 2010: 292-304
// example 4

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (i <= n0 + 1)

// tools

#include <stdlib.h>
#include <stdio.h>

int flip(void) {
	return rand() % 2;
}
#define OR(t1, t2) {if (flip()) {t1} else {t2}}
#define LOOP(j) {while (flip()) {j}}

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
#define CHECK(c)
#define CHECK_NOT(c)
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#define CHECK(c) {if (!(c)) checking_error();}
#define CHECK_NOT(c) {if (c) checking_error();}
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#ifdef GOOD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK(GOOD);}
#else
#ifdef BAD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK_NOT(BAD);}
#endif
#endif

// control and commands

#define S1 CONTROL(s == 1 && f == 1 && n > 0)
#define S2 CONTROL(s == 2 && f == 0 && n > 0)
#define S3 CONTROL(s == 2 && f == 1 && n > 0)
#define S4 CONTROL(s == 2 && f == 1 && n == 0)
#define S5 CONTROL(s == 1 && f == 0 && n > 0)
#define S6 CONTROL(s == 1 && f == 1 && n == 0)
#define S7 CONTROL(s == 2 && f == 0 && n == 0)
#define S8 CONTROL(s == 1 && f == 0 && n == 0)

#define G1 (s == 1 && f == 1)
#define A1 {s = 2; f = 0; i++;}
#define C1 COMMAND(G1, A1)

#define G2 (n > 0 && s == 2)
#define G2a (G2 && n > 1)
#define G2b (G2 && n == 1)
#define A2 {n--; f = 1; i++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define A3 {s = 1;}
#define C3 COMMAND(G3, A3)

#define INI {n0 = rand(); n = n0; f = 1; s = 1; i = 0;}

// transition system

void ts_singlestate(void) {
	int n0, n, f, s, i;
	INI;
	LOOP(OR(C1, OR(C2, C3)));
}

void ts_restructured(void) {
	int n0, n, f, s, i;
	INI;
	if (n > 0) {
		S1; C1; S2;
		OR(
			C2a; S3; LOOP(C2a; S3); C2b; S4,
		OR(
			C2b; S4,
			
			C3; S5
		))
	}
	else {
		S6; C1; S7; C3; S8
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

