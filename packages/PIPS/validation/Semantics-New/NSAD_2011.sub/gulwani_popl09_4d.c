// Sumit Gulwani, Krishna K. Mehra, Trishul M. Chilimbi: SPEED: precise and
// efficient static estimation of program computational complexity.
// POPL 2009: 127-139
// figure 4d

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (c1 <= n && c2 <= m)

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
#define S2 CONTROL(s == 1 && x >= n)
#define S3 CONTROL(s == 2 && x < n && y < m)
#define S4 CONTROL(s == 2 && x < n && y >= m)
#define S5 CONTROL(s == 2 && x >= n && y < m)
#define S6 CONTROL(s == 2 && x >= n && y >= m)

#define G1 (s == 1 && x < n)
#define G1a (s == 1 && x < n - 1 && 0 < m)
#define G1b (s == 1 && x < n - 1 && 0 >= m)
#define G1c (s == 1 && x == n - 1 && 0 < m)
#define G1d (s == 1 && x == n - 1 && 0 >= m)
#define A1 {s = 2; y = 0; x++; c1++; c2 = 0;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)
#define C1d COMMAND(G1d, A1)

#define G2 (s == 2 && y < m)
#define G2a (s == 2 && y < m - 1)
#define G2b (s == 2 && y == m - 1)
#define A2 {y++; c2++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define A3 {s = 1;}
#define C3 COMMAND(G3, A3)

#define INI {\
	n = rand(); m = rand();\
	x = c1 = c2 = 0;\
	y = rand();\
	s = 1;\
}

// transition system

void ts_singlestate(void) {
	int n, m, x, y, s, c1, c2;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int n, m, x, y, s, c1, c2;
	INI; CHECK;
	if (x < n) {
		S1;
		LOOP(
			OR(
				C1a; S3; LOOP(C2a; S3); C3; S1,
			OR(
				C1b; S4; C3; S1,
				
				C1a; S3; LOOP(C2a; S3); C2b; S4; C3; S1
			))
		)
		OR(
			C1c; S5; LOOP(C2a; S5);
			OR(C2b; S6; C3; S2, C3; S2),
			
			C1d; S6; C3; S2
		)
	}
	else {
		S2;
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

