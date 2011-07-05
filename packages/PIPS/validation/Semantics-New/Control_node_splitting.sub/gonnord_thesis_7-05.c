// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 7.5

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (t >= 0 && 0 <= s && s <= 4 && 0 <= d && d <= 4 * t + s)

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

#define S1 CONTROL(s <= 3)
#define S2 CONTROL(s > 3)

#define G1 (s <= 3)
#define G1a (s <= 2)
#define G1b (s == 3)
#define A1 {d++; s++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (1)
#define A2 {s = 0; t++;}
#define C2 COMMAND(G2, A2)

#define INI {s = t = d = 0;}

// transition system

void ts_singlestate(void) {
	int s, t, d;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int s, t, d;
	INI; CHECK;
	S1;
	LOOP(OR(C1a; S1, C1b; S2; C2; S1))
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

