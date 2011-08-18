// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 7.1

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (x + y0 == x0 + y)

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

#define S1 CONTROL(x + y <= -1 && z >= 0)
#define S2 CONTROL(x + y > - 1 && z >= 0)
#define S3 CONTROL(x + y <= -1 && z < 0)
#define S4 CONTROL(x + y > - 1 && z < 0)

#define G1 (x + y <= -1 && z >= 0)
#define G1a (x + y <= -3 && G1)
#define G1b (x + y > - 3 && G1)
#define A1 {b = 1; c = 1; x++; y++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (x + y <= -1 && z < 0)
#define A2 {b = 1; c = 0; x++; y--;}
#define C2 COMMAND(G2, A2)

#define G3 (x + y > - 1)
#define G3a (x + y > 1)
#define G3b (x + y <= 1 && G3)
#define A3 {b = 0; c = 0; x--; y--;}
#define C3 COMMAND(G3, A3)
#define C3a COMMAND(G3a, A3)
#define C3b COMMAND(G3b, A3)

#define INI {x0 = rand(); y0 = rand(); z = rand(); x = x0; y = y0;}

// transition system

void ts_singlestate(void) {
	int b, c, x0, y0, x, y, z;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int b, c, x0, y0, x, y, z;
	INI; CHECK;
	if (z >= 0) {
		if (x + y <= -1) {
			goto L1;
		}
		else {
			goto L2;
		}
	}
	else {
		if (x + y <= -1) {
			goto L3;
		}
		else {
			goto L4;
		}
	}
L1:
	S1; LOOP(C1a; S1); C1b; goto L2;
L2:
	S2; LOOP(C3a; S2); C3b; goto L1;
L3:
	S3; LOOP(C2; S3);
L4:
	S4; LOOP(C3a; S4); C3b; goto L3;
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

