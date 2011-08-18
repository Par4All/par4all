// Sumit Gulwani, Krishna K. Mehra, Trishul M. Chilimbi: SPEED: precise and
// efficient static estimation of program computational complexity.
// POPL 2009: 127-139
// figure 3d

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (c <= n)

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

#define S1 CONTROL(x < n && y < m)
#define S2 CONTROL(x < n && y >= m)
#define S3 CONTROL(x >= n && y < m)
#define S4 CONTROL(x >= n && y >= m)

#define G1 (x < n)
#define G1a (x < n - 1 && y < m - 1)
#define G1b (x < n - 1 && y == m - 1)
#define G1c (x == n - 1 && y < m - 1)
#define G1d (x == n - 1 && y == m - 1)
#define G1e (x < n - 1 && y >= m)
#define G1f (x == n - 1 && y >= m)
#define A1 {x++; y++; c++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)
#define C1d COMMAND(G1d, A1)
#define C1e COMMAND(G1e, A1)
#define C1f COMMAND(G1f, A1)

#define G2 (y < m)
#define G2a (x < n - 1 && y < m - 1)
#define G2b (x < n - 1 && y == m - 1)
#define G2c (x == n - 1 && y < m - 1)
#define G2d (x == n - 1 && y == m - 1)
#define G2e (x >= n && y < m - 1)
#define G2f (x >= n && y == m - 1)
#define A2 {x++; y++; c++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)
#define C2d COMMAND(G2d, A2)
#define C2e COMMAND(G2e, A2)
#define C2f COMMAND(G2f, A2)

#define INI {\
	n = rand(); m = rand(); ASSUME(n >= m);\
	x = y = c = 0;\
}

// transition system

void ts_singlestate(void) {
	int n, m, x, y, c;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int n, m, x, y, c;
	INI; CHECK;
	if (x < n)
		if (y < m) goto L1;
		else goto L2;
	else
		if (y < m) goto L3;
		else goto L4;
L1:
	S1; LOOP(OR(C1a; S1, C2a; S1));
	OR(
		OR(C1b, C2b); goto L2;,
	OR(
		OR(C1c, C2c); goto L3;,

		OR(C1d, C2d); goto L4;
	))
L2:
	S2; LOOP(C1e; S2); C1f; goto L4;
L3:
	S3; LOOP(C2e; S3); C2f; goto L4;
L4:
	S4;
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

