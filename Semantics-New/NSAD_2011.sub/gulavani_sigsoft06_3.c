// Thomas A. Henzinger, Ranjit Jhala, Rupak Majumdar, Grégoire Sutre: Lazy
// abstraction. POPL 2002:58-70
// figure 1
// Bhargav S. Gulavani, Thomas A. Henzinger, Yamini Kannan, Aditya V. Nori,
// Sriram K. Rajamani: SYNERGY: a new algorithm for property checking. SIGSOFT
// FSE 2006:117-127
// figure 3

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (s == 2 && lock == 0)

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

#define S1 CONTROL(s == 1 && x < y)
#define S2 CONTROL(s == 1 && x > y)
#define S3 CONTROL(s == 1 && x == y)
#define S4 CONTROL(s == 2)

#define G1 (s == 1 && x != y)
#define A1 {lock = 1; x = y;}
#define C1 COMMAND(G1, A1)

#define G2 (s == 1 && x != y)
#define A2 {lock = 0; x = y; y++;}
#define C2 COMMAND(G2, A2)

#define G3 (s == 1 && x == y)
#define A3 {s = 2;}
#define C3 COMMAND(G3, A3)

#define INI {lock = 0; x = rand(); y = rand(); ASSUME(x != y); s = 1;}

// transition system

void ts_singlestate(void) {
	int s, lock, x, y;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int s, lock, x, y;
	INI; CHECK;
	if (x < y) goto L1;
	else goto L2;
L1:
	S1; LOOP(C2; S1);
	C1; goto L3;
L2:
	OR(C2; goto L1;, C1; goto L3;);
L3:
	S3; C3; goto L4;
L4:
	S4;
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

