// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 7.7

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (0 <= l && l <= t && 6 * l <= t + 50)

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

#define S1 CONTROL(u <= 59 && v <= 9)
#define S2 CONTROL(u <= 59 && v > 9)
#define S3 CONTROL(u == 60)

#define G1 (u <= 59 && v <= 9)
#define G1a (u <= 58 && v <= 8)
#define G1b (u <= 58 && v == 9)
#define G1c (u == 59 && v <= 9)
#define A1 {u++; t++; l++; v++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)

#define G2 (u <= 59)
#define G2a (u <= 58)
#define G2b (u == 59)
#define A2 {u++; t++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (u == 60)
#define A3 {u = 0; v = 0;}
#define C3 COMMAND(G3, A3)

#define INI {u = 0; t = 0; l = 0; v = 0;}

// trannsition system

void ts_singlestate(void) {
	int u, t, l, v;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int u, t, l, v;
	INI; CHECK;
	S1;
	LOOP(
		OR(
			C1a; S1,
		OR(
			C2a; S1,
		OR(
			C1c; S3; C3; S1,
		OR(
			C2b; S3; C3; S1,
			
			C1b; S2; LOOP(C2a; S2); C2b; S3; C3; S1
		))))
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

