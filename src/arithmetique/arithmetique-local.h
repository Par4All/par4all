/* package arithmetique
 *
 * Francois Irigoin, mai 1989
 *
 * Modifications
 *  - reprise de DIVIDE qui etait faux (Remi Triolet, Francois Irigoin, 
 *    april 90)
 *  - simplification de POSITIVE_DIVIDE par suppression d'un modulo
 */

#ifndef ARITHMETIQUE_PACKAGE
#define ARITHMETIQUE_PACKAGE

/* valeur absolue */
#ifndef ABS
#define ABS(x) ((x)>=0 ? (x) : -(x))
#endif

/* minimum et maximum 
 * if they are defined somewhere else, they are very likely 
 * to be defined the same way. Thus the previous def is not overwritten.
 */
#ifndef MIN
#define MIN(x,y) ((x)>=(y)?(y):(x))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>=(y)?(x):(y))
#endif

/* signe d'un entier: -1, 0 ou 1 */
#define SIGN(x) ((x)>0? 1 : ((x)==0? 0 : -1))

/* division avec reste toujours positif
 * basee sur les equations:
 * a/(-b) = - (a/b)
 * (-a)/b = - ((a+b-1)/b)
 * ou a et b sont des entiers positifs
 */
#define DIVIDE(x,y) ((y)>0? POSITIVE_DIVIDE(x,y) : \
		     -POSITIVE_DIVIDE((x),(-(y))))

/* division avec reste toujours positif quand y est positif: assert(y>=0) */
#define POSITIVE_DIVIDE(x,y) ((x)>0 ? (x)/(y) : - (-(x)+(y)-1)/(y))

/* modulo a resultat toujours positif */
#define MODULO(x,y) ((y)>0 ? POSITIVE_MODULO(x,y) : POSITIVE_MODULO(-x,-y))

/* modulo par rapport a un nombre positif: assert(y>=0)
 *
 * Ce n'est pas la macro la plus efficace que j'aie jamais ecrite: il faut
 * faire, dans le pire des cas, deux appels a la routine .rem, qui n'est
 * surement pas plus cablee que la division ou la multiplication
 */
#define POSITIVE_MODULO(x,y) ((x) > 0 ? (x)%(y) : \
			      ((x)%(y) == 0 ? 0 : ((y)-(-(x))%(y))))
			      
/* Pour la recherche de performance, selection d'une implementation
 * particuliere des fonctions
 */

#define pgcd(a,b) pgcd_slow(a,b)

#define divide(a,b) DIVIDE(a,b)

#define modulo(a,b) MODULO(a,b)

typedef struct fract{long num, den, numero ; } frac ;

typedef struct col{int taille, existe ; frac *colonne ;} tableau ;

#endif /* ARITHMETIQUE_PACKAGE */
