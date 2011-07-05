// Sumit Gulwani, Florian Zuleger: The reachability-bound problem.
// PLDI 2010: 292-304
// example 6

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (i1 <= n0 - x0 && i2 <= n0 - z0)

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

#define S1 CONTROL(x < n && z > x)
#define S2 CONTROL(x < n && z <= x)
#define S3 CONTROL(x >= n)

#define G1 (x < n && z > x)
#define G1a (x < n - 1 && z > x + 1)
#define G1b (x < n - 1 && z == x + 1)
#define G1c (x == n - 1 && z > x)
#define A1 {x++; i1++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)

#define G2 (x < n && z <= x)
#define G2a (x < n && z < x)
#define G2b (x < n && z == x)
#define A2 {z++; i2++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define INI {\
	n0 = rand(); x0 = rand(); z0 = rand(); ASSUME(n0 >= x0 && n0 >= z0);\
	n = n0; x = x0; z = z0;\
	i1 = i2 = 0;\
}

// transition system

void ts_singlestate(void) {
	int n0, x0, z0, n, x, z, i1, i2;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int n0, x0, z0, n, x, z, i1, i2;
	INI; CHECK;
	if (x < n) {
		if (z <= x) {
			S2; LOOP(C2a; S2); C2b;
		}
		S1;
		LOOP(
			OR(
				C1a; S1,
				
				C1b; S2;
				LOOP(C2a; S2);
				C2b; S1
			)
		)
		C1c;
	}
	S3
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

