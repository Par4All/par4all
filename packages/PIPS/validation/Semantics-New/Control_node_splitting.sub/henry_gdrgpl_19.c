// Julien Henry: Static Analysis by Abstract Interpretation, Path Focusing
// talk at GDR GPL, 8/06/2011

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (-1 <= d && d <= 1 && d >= 2 * x - 1999)

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

#define S1 CONTROL(x == 0 && d == 1)
#define S2 CONTROL(1 <= x && x <= 999 && d == 1)
#define S3 CONTROL(x == 1000 && d == 1)
#define S4 CONTROL(1 <= x && x <= 999 && d == -1)
#define S5 CONTROL(x == 0 && d == -1)

#define G1 (x == 0)
#define A1 {d = 1; x += d;}
#define C1 COMMAND(G1, A1)

#define G2 (1 <= x && x <= 999)
#define G2a (1 <= x && x <= 998)
#define G2b (x == 999)
#define G2c (2 <= x && x <= 999)
#define G2d (x == 1)
#define A2 {x += d;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)
#define C2d COMMAND(G2d, A2)

#define G3 (x == 1000)
#define A3 {d = -1; x += d;}
#define C3 COMMAND(G3, A3)

#define INI {\
	x = 0; \
	d = 1; \
}

// transition system

void ts_singlestate(void) {
	int x, d;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}
		
void ts_restructured(void) {
	int x, d;
	INI; CHECK;
	S1; C1;
	LOOP(
		S2; LOOP(C2a; S2);
		C2b; S3;
		C3; S4;
		LOOP(C2c; S4);
		C2d; S5;
		C1;
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

