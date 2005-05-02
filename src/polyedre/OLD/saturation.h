/* saturation.h
 * structures de donnees utilises pour les calculs de saturations
 * par les operateurs sur les polyedres, en particulier les intersections
 * de polyedre avec des hyperplans ou demi-espaces
 *
 * Francois Irigoin, 25 mai 1989
 */

#define SAT_TAB 9
#define TSATX 10 /* une unique utilisation avec MALLOC */
#define TINT    12/* une utilisation avec MALLOC */

/* fonctions utilisant ces structures de donnees */

/* int satur_som(Psommet s, Pcontrainte eq): saturation d'une contrainte eq
 * (implicitement une inegalite) par un sommet s
 *
 * D'apres Halbwachs page 9 section 1.3.1, un point X (cas general d'un sommet)
 * sature une contrainte si A . X = B
 *
 * Ici, on calcule A. X*denominateur(X) - B*denominateur(X)
 *
 * Il faut donc tester satur_som()==0 pour savoir si le sommet ad_som
 * sature l'inegalite eq.
 *
 * il faut tester satur_som <= 0 pour savoir si le sommet s verifie
 * l'inegalite eq
 */
int satur_som();

/* int satur_vect(Pvecteur r, Pcontrainte eq): saturation d'une contrainte eq
 * (implicitement une inegalite)
 * par un rayon r (represente ici uniquement par son vecteur)
 *
 * D'apres Halbwachs page 9 section 1.3.1, un rayon r sature une contrainte
 * si A . r = 0
 *
 * Le test de saturation est donc satur_vect(r,eq)==0
 *
 * Le rayon r respecte l'inegalite eq si satur_vect(r,eq) <= 0 car les
 * inegalites sont mises sous la forme A . X <= B
 */
extern int satur_vect();

/* void ad_s_eq_sat(Psommet ns, Pcontrainte eq_vues, int nb_vues, int nb_eq):
 *
 * on a un nouveau sommet ns pour lequel on veut calculer les nb_vues
 * premieres saturations pour les premieres equations de la liste eq_vues
 * le resultat est dans un tableau (ns->eq_sat) ou la valeur de saturation
 * dans une case d'indice i est associee a l'equation de rang i dans la
 * liste donnee
 *
 * nb_eq est le nombre total d'equation de la liste, l'allocation de eq_sat
 * est faite avec un tel nombre car il se peut que ce sommet soit
 * conserve et que de nouvelles saturation soient a calculer pour lui.
 * on ne calcule pas non plus toutes les saturations car ce sommet peut
 * etre detruit sans les utiliser
 */
extern void ad_s_eq_sat();

/* void ad_r_eq_sat(Pray_dte nr, Pcontrainte eq_vues, int nb_vues, int nb_eq):
 *
 * on a un nouveau rayon nr pour lequel on veut calculer les nb_vues
 * premieres saturations pour les premieres equations de la liste eq_vues
 * le resultat est dans un tableau (nr->eq_sat) ou la valeur de saturation
 * dans une case d'indice i est associee a l'equation de rang i dans la
 * liste donnee
 *
 * nb_eq est le nombre total d'equation de la liste, l'allocation de eq_sat
 * est faite avec un tel nombre car il se peut que ce sommet soit
 * conserve et que de nouvelles saturation soient a calculer pour lui.
 * on ne calcule pas non plus toutes les saturations car ce sommet peut
 * etre detruit sans les utiliser
 */
extern void ad_r_eq_sat();

/* void d_sat_s_inter(Ps_sat_inter tab_sat, int nb_s, Psommet lsoms,
 *                    Pcontrainte eq, int num_eq):
 * initialisation de la structure de donnees locale a l'intersection de
 * SG et Demi-espace, on calcule les saturations afin de savoir les
 * combinaisons a faire
 *
 * on n'oublie pas d'affecter le champ eq_sat des sommets
 * pour l'elimination de redondance
 */
extern void d_sat_s_inter();

/* void d_sat_vect_inter(Ps_sat_inter tab_sat, int nb_vect, Pray_dte list_v,
 *                    Pcontrainte eq, int num_eq):
 * initialisation de la structure de donnees locale a l'intersection de
 * SG et Demi-espace, on calcule les saturations afin de savoir les
 * combinaisons a faire
 *
 * on n'oublie pas d'affecter le champ eq_sat des sommets
 * pour l'elimination de redondance
 */
void d_sat_vect_inter();

/* void sat_s_inter(Ps_sat_inter tab_sat, int nb_s, Psommet lsoms, 
 *                  Pcontrainte eq):
 * meme usage que les fonctions precedentes sauf qu'ici on n'utilise pas
 *	l'elimination de redondance -> intersection avec un hyperplan
 *	(rappel : on n'elimine pas les redondances sur les egalites)
 */
void sat_s_inter();

/* void sat_vect_inter(Pvect_sat_inter tab_sat, int nb_vect, Pray_dte liste_v,
 *                     Pcontrainte eq):
 */
void sat_vect_inter();

/* void free_inter(Ps_sat_inter sat_ss
 *                 Pvect_sat_inter sat_rs,
 *                 Pvect_sat_inter sat_ds,
 *                 int nb_ss,
 *                 int nb_rs,
 *                 int nb_ds): 
 * elimination d'elements generateurs devenus inutiles
 * dans l'intersection d'un polyedre avec un hyperplan
 */
void free_inter();

/* void free_demi_inter(Ps_sat_inter sat_ss,
 *                      Pvect_sat_inter sat_rs,
 *                      int nb_ss,
 *                      int nb_rs):
 * elimination lors de l'intersection de SG avec un demi-espace
 */
void free_demi_inter();

/* void new_ss(Psommet ns, Ss_sat_inter sat_s1, Ss_sat_inter sat_s2):
 * construction de sommet pour inter-demi-espace par composition de 
 * sommets de saturations opposees
 */
void new_ss();

/* void new_s(Psommet ns, Ss_sat_inter sat_so, Svect_sat_inter sat_v):
 * construction de vecteur pour inter-demi-espace (ou hyperplan) 
 * par composition d'un sommet et d'un rayon ou d'une droite
 */
extern void new_s();

/* void new_rr(Pray_dte nr, Svect_sat_inter sat_r1, Svect_sat_inter sat_r2):
 * construction de rayon pour inter-demi-espace par composition de 
 * rayons de saturations opposees
 */
extern void new_rr();
