/* 
 * $Id$
 *
 * This file contains the main for pips.
 * Please, do not change anything! do any change to pips_main().
 *
 * FC.
 */

extern char * pips_thanks(char *, char *);
extern int pips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("pips", argv[0]);
    return pips_main(argc, argv);
}
