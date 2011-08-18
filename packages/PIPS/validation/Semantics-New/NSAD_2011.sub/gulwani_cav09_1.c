// Sumit Gulwani: SPEED: Symbolic Complexity Bound Analysis. CAV 2009:51-62
// figure 1

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (i <= 100 + m)

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

#define S1 CONTROL(x < 100 && y < m)
#define S2 CONTROL(x < 100 && y >= m)
#define S3 CONTROL(x >= 100 && y >= m)

#define G1 (x < 100 && y < m)
#define G1a (x < 100 && y < m - 1)
#define G1b (x < 100 && y == m - 1)
#define A1 {y++; i++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (x < 100 && y >= m)
#define G2a (x < 99 && y >= m)
#define G2b (x == 99 && y >= m)
#define A2 {x++; i++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define INI {x = y = i = 0; m = rand();}

// transition system

void ts_singlestate(void) {
	int x, y, m, i;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int x, y, m, i;
	INI; CHECK;
	if (y < m) {
		S1; LOOP(C1a; S1); C1b;
	}
	S2; LOOP(C2a; S2); C2b; S3
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

