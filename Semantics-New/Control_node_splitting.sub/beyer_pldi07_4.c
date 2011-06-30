// Dirk Beyer, Thomas A. Henzinger, Rupak Majumdar, Andrey Rybalchenko: Path
// invariants. PLDI 2007: 300-309
// figure 4

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (s == 2 && y != 100)

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

#define S1 CONTROL(s == 1 && x < 50)
#define S2 CONTROL(s == 1 && 50 <= x && x < 100)
#define S3 CONTROL(s == 1 && x >= 100)
#define S4 CONTROL(s == 2)

#define G1 (s == 1 && x < 50)
#define G1a (s == 1 && x < 49)
#define G1b (s == 1 && x == 49)
#define A1 {x++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (s == 1 && 50 <= x && x < 100)
#define G2a (s == 1 && 50 <= x && x < 99)
#define G2b (s == 1 && x == 99)
#define A2 {x++; y++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 1 && x >= 100)
#define A3 {s = 2;}
#define C3 COMMAND(G3, A3)

#define INI {s = 1; x = 0; y = 50;}

// transition system

void ts_singlestate(void) {
	int s, x, y;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int s, x, y;
	INI; CHECK;
	S1; LOOP(C1a; S1);
	C1b; S2; LOOP(C2a; S2);
	C2b; S3;
	C3; S4
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

