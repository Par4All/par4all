// Same as maisonneuve17-1, restructured with guards.

// $Id$

#define USE_ASSERT 1
#define USE_CHECK 1
#define BAD1 (x == 2 && y == 7)
#define BAD2 (x == 5 && y == 7)

// usefull stuff

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void assert_error(void) {
	exit(1);
}

void checking_error(void) {
	exit(2);
}

#if USE_ASSERT == 0
#define assert(e)
#else
#define assert(e) {if (!(e)) assert_error();}
#endif

#if USE_CHECK == 0
#define check(e)
#define check_not(e)
#else
#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#endif

#define freeze {while(1);}

#define trans(g, c) {if (g) {c; check_not(BAD1); check_not(BAD2);} else freeze;}

// transition system

#define G1 (x >= 0 && x <= 5 && y >= 0 && y <= 5)
#define U1 {x++; y++;}
#define T1 {trans(G1, U1);}

#define G2 (x >= 4 && x <= 9 && y >= 4 && y <= 9)
#define U2 {x--; y--;}
#define T2 {trans(G2, U2);}

#define T1a_1n2_n1n2 {trans(G1 && x == 5 && y <= 2, U1);}
#define T1b_1n2_n1n2 {trans(G1 && x <= 2 && y == 5, U1);}
#define T1a_1n2_1n2 {trans(G1 && x <= 2 && y <= 4, U1);}
#define T1b_1n2_1n2 {trans(G1 && x <= 4 && y <= 2, U1);}
#define T1_1n2_12 {trans(G1 && x >= 3 && x <= 4 && y >= 3 && y <= 4 && y <= 7 - x, U1);}
#define T1_12_12 {trans(G1 && x == 4 && y == 4, U1);}
#define T1_12_n12 {trans(G1 && y >= 9 - x, U1);}
#define T1a_1n2_n12 {trans(G1 && x == 3 && y == 5, U1);}
#define T1b_1n2_n12 {trans(G1 && x == 5 && y == 3, U1);}

#define T2a_n12_n1n2 {trans(G2 && x == 4 && y >= 7, U2);}
#define T2b_n12_n1n2 {trans(G2 && x >= 7 && y == 4, U2);}
#define T2a_n12_n12 {trans(G2 && x >= 5 && y >= 7, U2);}
#define T2b_n12_n12 {trans(G2 && x >= 7 && y >= 5, U2);}
#define T2_n12_12 {trans(G2 && x >= 5 && x <= 6 && y >= 5 && y <= 6 && y >= 11 - x, U2);}
#define T2_12_12 {trans(G2 && x == 5 && y == 5, U2);}
#define T2_12_1n2 {trans(G2 && y <= 9 - x, U2);}
#define T2a_n12_1n2 {trans(G2 && x == 4 && y == 6, U2);}
#define T2b_n12_1n2 {trans(G2 && x == 6 && y == 4, U2);}

#define S_12 {assert(G1 && G2);}
#define S_1n2 {assert(G1 && !G2);}
#define S_n12 {assert(!G1 && G2);}
#define S_n1n2 {"deadlock"; assert(!G1 && !G2);}

#define LOOP_12 {\
	while (flip()) {\
		if (flip()) {\
			T1_12_12; S_12;\
		}\
		else {\
			T2_12_12; S_12;\
		}\
	}\
}

#define LOOP_1n2 {\
	while (flip()) {\
		if (flip()) {\
			T1a_1n2_1n2; S_1n2;\
		}\
		else if (flip()) {\
			T1b_1n2_1n2; S_1n2;\
		}\
	}\
}
	
#define LOOP_n12 {\
	while (flip()) {\
		if (flip()) {\
			T2a_n12_n12; S_n12;\
		}\
		else {\
			T2b_n12_n12; S_n12;\
		}\
	}\
}

#define TRANS_1n2_n1n2 {\
	if (flip()) {\
		T1a_1n2_n1n2; S_n1n2;\
	}\
	else if (flip()) {\
		T1b_1n2_n1n2; S_n1n2;\
	}\
}
	
#define TRANS_n12_n1n2 {\
	if (flip()) {\
		T2a_n12_n1n2; S_n1n2; freeze;\
	}\
	else if (flip()) {\
		T2b_n12_n1n2; S_n1n2; freeze;\
	}\
}
	
#define TRANS_1n2_12 {\
	T1_1n2_12; S_12;\
}

#define TRANS_n12_12 {\
	T2_n12_12; S_12;\
}

#define TRANS_12_1n2 {\
	T2_12_1n2; S_1n2;\
}

#define TRANS_12_n12 {\
	T1_12_n12; S_n12;\
}

#define TRANS_1n2_n12 {\
	if (flip()) {\
		T1a_1n2_n12;\
	}\
	else {\
		T1b_1n2_n12;\
	}\
}

#define TRANS_n12_1n2 {\
	if (flip()) {\
		T2a_n12_1n2;\
	}\
	else {\
		T2b_n12_1n2;\
	}\
}

void run(void) {
	
	int x, y;
	x = rand();
	y = rand();
	if (G1 && y <= 5 - x) {
		
		while (1) {
			S_1n2;
			if (flip()) {
				TRANS_1n2_n1n2;
			}
			else if (flip()) {
				LOOP_1n2;
			}
			else if (flip()) {
				TRANS_1n2_12;
				LOOP_12;
				TRANS_12_1n2;
			}
			else {
				if (flip()) {
					TRANS_1n2_n12;
				}
				else {
					TRANS_1n2_12;
					LOOP_12;
					TRANS_12_n12;
				}
				while (flip()) {
					if (flip()) {
						LOOP_n12;
					}
					else {
						TRANS_n12_12;
						LOOP_12;
						TRANS_12_n12;
					}
				}
				if (flip()) {
					TRANS_n12_n1n2;
				}
				else if (flip()) {
					TRANS_n12_1n2;
				}
				else {
					TRANS_n12_12;
					LOOP_12;
					TRANS_12_1n2;
				}
			}
		}
		
	}	
	
}

int main(void) {
	run();
	return 0;
}

