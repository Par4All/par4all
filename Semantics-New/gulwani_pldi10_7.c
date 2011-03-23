// Sumit Gulwani, Florian Zuleger: The reachability-bound problem.
// PLDI 2010: 292-304
// example 7

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (i <= m + 1)

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

#define S1 CONTROL(n < j && j <= m)
#define S2 CONTROL(j > m)
#define S3 CONTROL(j < n)
#define S4 CONTROL(j == n)

#define G1 (j != n && j <= m)
#define G1a (j != n && j < m)
#define G1b (j != n && j == m)
#define G1c (j != n && j < n - 1)
#define G1d (j != n && j == n - 1)
#define A1 {j++; i++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)
#define C1d COMMAND(G1d, A1)

#define G2 (j != n && j > m)
#define G2a (G2 && 0 < n)
#define G2b (G2 && 0 == n)
#define A2 {j = 0; i++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define INI {n = rand(); m = rand(); ASSUME(0 < n && n < m); j = n + 1; i = 0;}

// transition system

void ts_singlestate(void) {
	int n, j, m, i;
	INI;
	LOOP(OR(C1, C2));
}

void ts_restructured(void) {
	int n, j, m, i;
	INI;
	if (j <= m) {
		S1; LOOP(C1a; S1); C1b;
	}
	S2;
	OR(
		C2b; S4,
		
		C2a; S3; LOOP(C1c; S3); C1d; S4
	);
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

