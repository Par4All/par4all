// Sumit Gulwani, Sagar Jain, Eric Koskinen: Control-flow refinement and
// progress invariants for bound analysis. PLDI 2009: 375-385
// example 1
// Sumit Gulwani, Florian Zuleger: The reachability-bound problem.
// PLDI 2010: 292-304
// example 7

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (j <= n + 1)

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

#define S1 CONTROL(i < t && t <= n)
#define S2 CONTROL(t > n)
#define S3 CONTROL(t < i)
#define S4 CONTROL(t == i)

#define G1 (t != i && t <= n)
#define G1a (t != i && t < n)
#define G1b (t != i && t == n)
#define G1c (t != i && t < i - 1)
#define G1d (t != i && t == i - 1)
#define A1 {t++; j++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)
#define C1d COMMAND(G1d, A1)

#define G2 (t != i && t > n)
#define G2a (G2 && 0 < i)
#define G2b (G2 && 0 == i)
#define A2 {t = 0; j++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define INI {i = rand(); n = rand(); ASSUME(i <= n); t = i + 1; j = 0;}

// transition system

void ts_singlestate(void) {
	int i, t, n, j;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int i, t, n, j;
	INI; CHECK;
	if (t <= n) {
		S1; LOOP(C1a; S1); C1b;
	}
	S2;
	OR(
		C2b; S4,
		
		C2a; S3; LOOP(C1c; S3); C1d; S4
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

