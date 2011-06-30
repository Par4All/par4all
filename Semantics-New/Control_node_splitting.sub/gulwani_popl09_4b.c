// Sumit Gulwani, Krishna K. Mehra, Trishul M. Chilimbi: SPEED: precise and
// efficient static estimation of program computational complexity.
// POPL 2009: 127-139
// figure 4b

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (c1 <= n - x0 && c2 <= m - y0)

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

#define S1 CONTROL(s == 1 && x < n && y < m)
#define S2 CONTROL(s == 1 && x < n && y >= m)
#define S3 CONTROL(s == 1 && x >= n)
#define S4 CONTROL(s == 2 && x < n && y < m)
#define S5 CONTROL(s == 2 && x < n && y >= m)

#define G1 (s == 1 && x < n)
#define A1 {s = 2;}
#define C1 COMMAND(G1, A1)

#define G2 (s == 2 && y < m)
#define G2a (s == 2 && y < m - 1)
#define G2b (s == 2 && y == m - 1)
#define A2 {y++; c2++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2)
#define G3a (s == 2 && x < n - 1)
#define G3b (s == 2 && x == n - 1)
#define A3 {s = 1; x++; c1++; c2 = 0;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define INI {\
	x0 = rand(); y0 = rand(); x = x0; y = y0;\
	n = rand(); m = rand(); ASSUME(n >= x0 && m >= y0);\
	s = 1; c1 = c2 = 0;\
}

// transition system

void ts_singlestate(void) {
	int x0, y0, x, y, n, m, s, c1, c2;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int x0, y0, x, y, n, m, s, c1, c2;
	INI; CHECK;
	if (x < n)
		if (y < m) {
			S1; LOOP(C1; S4; LOOP(C2a; S4); C3a; S1);
			C1; S4; LOOP(C2a; S4);
			OR(C2b; S5; C3b; S3, C3b; S3);
		}
		else {
			S2; LOOP(C1; S5; C3a; S2);
			C1; S5;
			C3b; S3;
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

