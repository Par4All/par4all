/* 
 * $Id$
 *
 * This file contains the main for fpips.
 * Please, do not change anything! do any change to fpips_main().
 *
 * FC.
 */

extern char * pips_thanks(char *, char *);
extern int fpips_main(int, char**);

int main(int argc, char ** argv)
{
    pips_thanks("fpips", argv[0]);
    return fpips_main(argc, argv);
}
