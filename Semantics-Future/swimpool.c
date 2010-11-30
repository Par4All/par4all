// Swimming pool example from M. Latteux:
// * G.W. Brams. Reseaux de Petri: Theorie et Pratique. Masson, Paris, 1983.
// * R. David and H. Alla. Du Grafcet aux Reseaux de Petri, Hermes, Paris, 1989.
// Adapted from FAST library:
// * http://www.lsv.ens-cachan.fr/Publis/PAPERS/PS/BerFri-concur99.ps
// * http://www.lsv.ens-cachan.fr/Software/fast/examples/swimmingpool.fst

#include <stdlib.h>

void error() {
	exit(1);
}

int flip() {
	return rand() % 2;
}

#define guard(c) if (!(c)) return;

#define G1 (x6 >= 1)
#define C1 {x1++; x6--;}
#define T1 {guard(G1); C1;}
#define G2 (x1 >= 1 && x7 >= 1)
#define C2 {x2++; x1--; x7--;}
#define T2 {guard(G2); C2;}
#define G3 (x2 >= 1)
#define C3 {x6++; x3++; x2--;}
#define T3 {guard(G3); C3;}
#define G4 (x3 >= 1 && x6 >= 1)
#define C4 {x4++; x3--; x6--;}
#define T4 {guard(G4); C4;}
#define G5 (x4 >= 1)
#define C5 {x5++; x7++; x4--;}
#define T5 {guard(G5); C5;}
#define G6 (x6 >= 1)
#define C6 {x6++; x5--;}
#define T6 {guard(G6); C6;}

/*#define chk_ok {if (x1 + x2 + x4 + x5 + x6 == 0) error();}*/
#define chk_ok {if (x1 == 0 && x2 == 0 && x4 == 0 && 5 == 0 && x6 == 0) error();}
/*#define chk_ok {if (x1 == 0 && x2 == 0 && x4 == 0 && 5 == 0 && x6 == 0 && (x1 == 0 || x7 == 0) && (x3 == 0 || x6 == 0)) error();}*/

void run(void) {
	
	int x1, x2, x3, x4, x5, x6, x7;
	x1 = x2 = x3 = x4 = x5 = 0;
	x6 = rand();
	x7 = rand();

	if (x6 >= 1 && x7 >= 0) {
		while (1) {
			while (flip()) T1;
			chk_ok;
			while (flip()) T2;
			chk_ok;
			while (flip()) T3;
			chk_ok;
			while (flip()) T4;
			chk_ok;
			while (flip()) T5;
			chk_ok;
			while (flip()) T6;
			chk_ok;
		}
	}
	
}

int main(void) {
	run();
	return 0;
}

