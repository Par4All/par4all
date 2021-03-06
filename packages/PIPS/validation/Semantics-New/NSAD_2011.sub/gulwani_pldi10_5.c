// Sumit Gulwani, Florian Zuleger: The reachability-bound problem.
// PLDI 2010: 292-304
// example 5

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (j <= n0)

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

#define S1 CONTROL(i < n && f == 0 && s == 1)
#define S2 CONTROL(i < n && f == 1 && s == 1)
#define S3 CONTROL(i >= n && s == 1)
#define S4 CONTROL(i < n && f == 0 && s == 2)
#define S5 CONTROL(i < n && f == 1 && s == 2)
#define S6 CONTROL(i >= n && s == 2)

#define G1 (i < n && s == 1)
#define A1 {f = 0; s = 2; j++;}
#define C1 COMMAND(G1, A1)

#define G2 (f == 1 && s == 2)
#define G2a (G2 && i < n - 1)
#define G2b (G2 && i == n - 1)
#define A2 {n--; j++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (f == 0 && s == 2)
#define G3a (G3 && i < n - 1)
#define G3b (G3 && i == n - 1)
#define A3 {i++; s = 1;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define G4 (f == 1 && s == 2)
#define A4 {s = 1;}
#define C4 COMMAND(G4, A4)

#define INI {n0 = rand(); n = n0; i = 0; f = rand(); ASSUME(f <= 1); j = 0; s = 1;}

// transition system

void ts_singlestate(void) {
	int n0, n, i, f, j, s;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, C4))))
}

void ts_restructured(void) {
	int n0, n, i, f, j, s;
	INI; CHECK;
	if (i < n) {
		if (f == 0) {
			S1; C1; S4;
		}
		else {
			S2; C1; S4;
		}
		LOOP(
			OR(
				C3a; S1; C1; S4,
				
				C2a; S5; C4; S2; C1; S4
			)
		)
		OR(
			C2b; S6; C4; S3,
			
			C3b; S3
		)
	}
	else {
		S3;
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

