// Sumit Gulwani: SPEED: Symbolic Complexity Bound Analysis. CAV 2009:51-62
// figure 3

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

#define S1 CONTROL(s == 1 && x < n)
#define S2 CONTROL(s == 1 && x >= n)
#define S3 CONTROL(s == 2 && y < n)
#define S4 CONTROL(s == 2 && y >= n)

#define G1 (s == 1 && x < n)
#define A1 {s = 2; i1++; y = x;}
#define C1 COMMAND(G1, A1)

#define G2 (s == 2 && y < n)
#define G2a (s == 2 && y < n - 1)
#define G2b (s == 2 && y == n - 1)
#define A2 {i2++; y++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define G3a (s == 2 && y + 1 < n)
#define G3b (s == 2 && y + 1 >= n)
#define A3 {s = 1; x = y + 1;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define INI {x = i1 = i2 = 0; s = 1; n = rand();}

// transition system

void ts_singlestate(void) {
	int s, x, y, n, i1, i2;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int s, x, y, n, i1, i2;
	INI; CHECK;
	if (x < n) {
		S1; C1; S3;
		LOOP(
			OR(
				C2a; S3,
				
				C3a; S1; C1; S3
			)
		)
		OR(
			C3b; S2,
			
			C2b; S4; C3; S2
		)
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

