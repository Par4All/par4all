// Sumit Gulwani, Krishna K. Mehra, Trishul M. Chilimbi: SPEED: precise and
// efficient static estimation of program computational complexity.
// POPL 2009: 127-139
// figure 3c

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (c <= n + 1)

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

#define S1 CONTROL(s == 1 && x < n)
#define S2 CONTROL(s == 2 && x < n)
#define S3 CONTROL(s == 1 && x >= n)
#define S4 CONTROL(s == 2 && x >= n)

#define G1 (s == 1 && x < n)
#define A1 {s = 2;}
#define C1 COMMAND(G1, A1)

#define G2 (s == 2 && x < n)
#define G2a (s == 2 && x < n - 1)
#define G2b (s == 2 && x == n - 1)
#define A2 {x++; c++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define G3a (s == 2 && x < n - 1)
#define G3b (s == 2 && x == n - 1)
#define A3 {s = 1; x++; c++;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define INI {n = rand(); x = c = 0; s = 1;}

// transition system

void ts_singlestate(void) {
	int n, x, c, s;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int n, x, c, s;
	INI; CHECK;
	if (x >= n) {
		S3;
	}
	else {
		S1; C1;
		S2; LOOP(OR(C2a; S2, C3a; S1; C1; S2));
		OR(C2b; S4; C3; S3, C3b; S3);
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

