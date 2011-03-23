// Denis Gopan, Thomas W. Reps: Guided Static Analysis. SAS 2007: 349-365

// $Id$

// parameters

#define DO_CONTROL 1
#define DO_CHECKING 1
#define GOOD (s == 2 || (y <= x && y <= -x + 102))

// tools

#include <stdlib.h>
#include <stdio.h>

int flip(void) {
	return rand() % 2;
}
#define OR(t1, t2) {if (flip()) {t1} else {t2}}
#define LOOP(t) {while (flip()) {t}}

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
#define CHECK(c)
#define CHECK_NOT(c)
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#define CHECK(c) {if (!(c)) checking_error();}
#define CHECK_NOT(c) {if (c) checking_error();}
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#ifdef GOOD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK(GOOD);}
#else
#ifdef BAD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK_NOT(BAD);}
#endif
#endif

// control and commands

#define S1 CONTROL(s == 1 && x <= 50 && y >= 0)
#define S2 CONTROL(s == 2 && x <= 50 && y >= 0)
#define S3 CONTROL(s == 1 && x > 50 && y >= 0)
#define S4 CONTROL(s == 2 && x > 50 && y >= 0)
#define S5 CONTROL(s == 2 && x > 50 && y < 0)

#define G1 (s == 1 && x <= 50)
#define A1 {s = 2; y++;}
#define C1 COMMAND(G1, A1)

#define G2 (s == 1 && x > 50)
#define G2a (G2 && y == 0)
#define G2b (G2 && y >= 1)
#define A2 {s = 2; y--;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 2 && y >= 0)
#define G3a (G3 && x <= 49)
#define G3b (G3 && x == 50)
#define A3 {s = 1; x++;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define INI {s = 1; x = y = 0;}

// transition system

void ts_singlestate(void) {
	int s, x, y;
	INI;
	LOOP(OR(C1, OR(C2, C3)));
}

void ts_restructured(void) {
	int s, x, y;
	INI;
	S1;
	LOOP(C1; S2; C3a; S1);
	C1; S2; C3b; S3;
	LOOP(C2b; S4; C3; S3);
	C2a; S5;
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

