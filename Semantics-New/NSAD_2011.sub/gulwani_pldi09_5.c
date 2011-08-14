// Sumit Gulwani, Sagar Jain, Eric Koskinen: Control-flow refinement and
// progress invariants for bound analysis. PLDI 2009: 375-385
// example 5

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (j1 <= n - m && j2 <= m && j1 * j2 == 0)

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

#define S1 CONTROL(0 < i && i < n && b == 1)
#define S2 CONTROL(0 < i && i < n && b == 0)
#define S3 CONTROL(i >= n)
#define S4 CONTROL(i <= 0)

#define G1 (0 < i && i < n && b == 1)
#define G1a (G1 && i < n - 1)
#define G1b (G1 && i == n - 1)
#define A1 {i++; j1++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (0 < i && i < n && b == 0)
#define G2a (G2 && 1 < i)
#define G2b (G2 && 1 == i)
#define A2 {i--; j2++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define INI {m = rand(); n = rand(); b = rand(); ASSUME(0 < m && m < n && b <= 1); i = m; j1 = j2 = 0;}

// transition system

void ts_singlestate(void) {
	int i, m, n, b, j1, j2;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int i, m, n, b, j1, j2;
	INI; CHECK;
	if (b == 1) {
		S1; LOOP(C1a; S1); C1b; S3;
	}
	else {
		S2; LOOP(C2a; S2); C2b; S4;
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

