// Sumit Gulwani, Sagar Jain, Eric Koskinen: Control-flow refinement and
// progress invariants for bound analysis. PLDI 2009: 375-385
// example 3

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (k1 <= m && k2 <= n)

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

#define S1 CONTROL(i < n && j < m)
#define S2 CONTROL(i < n && j >= m)
#define S3 CONTROL(i >= n)

#define G1 (i < n && j < m)
#define G1a (i < n && j < m - 1)
#define G1b (i < n && j == m - 1)
#define A1 {j++; k1++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (i < n && j >= m)
#define G2a (i < n - 1 && j >= m && 0 < m)
#define G2b (i < n - 1 && j >= m && 0 >= m)
#define G2c (i == n - 1 && j >= m)
#define A2 {j = 0; i++; k1 = 0; k2++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)

#define INI {i = j = k1 = k2 = 0; m = rand(); n = rand();}

// transition system

void ts_singlestate(void) {
	int i, j, m, n, k1, k2;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int i, j, m, n, k1, k2;
	INI; CHECK;
	if (i < n)
		if (j < m) goto L1;
		else goto L2;
	else goto L3;
L1:
	S1; LOOP(C1a; S1); C1b;
L2:
	S2;
	LOOP(
		OR(
			C2b; S2,
			
			C2a; S1; LOOP(C1a; S1); C1b; S2
		)
	)
	C2c;
L3:
	S3;
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

