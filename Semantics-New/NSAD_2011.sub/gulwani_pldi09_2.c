// Sumit Gulwani, Sagar Jain, Eric Koskinen: Control-flow refinement and
// progress invariants for bound analysis. PLDI 2009: 375-385
// example 2

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (i1 <= n && i2 <= n)

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

#define S1 CONTROL(v1 > 0 && v2 < m)
#define S2 CONTROL(v1 > 0 && v2 >= m)
#define S3 CONTROL(v1 <= 0)

#define G1 (v1 > 0 && v2 < m)
#define G1a (v1 > 1 && v2 < m - 1)
#define G1b (v1 > 1 && v2 == m - 1)
#define G1c (v1 == 1 && v2 < m)
#define A1 {v2++; v1--; i1++; i2 = 0;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)

#define G2 (v1 > 0 && v2 >= m)
#define A2 {v2 = 0; i2 += m;}
#define C2 COMMAND(G2, A2)

#define INI {n = rand(); m = rand(); ASSUME(n > 0 && m > 0); v1 = n; v2 = 0; i1 = i2 = 0;}

// transition system

void ts_singlestate(void) {
	int v1, v2, n, m, i1, i2;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int v1, v2, n, m, i1, i2;
	INI; CHECK;
	S1;
	LOOP(
		OR(
			C1a; S1,
			
			C1b; S2; C2; S1
		)
	)
	C1c; S3
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

