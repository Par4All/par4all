/* Dealing with signed/unsigned conversions
 *
 * The initial idea to handle this was to delay the conversion from
 * signed to unsigned when prettyprinting transformers and
 * preconditions.
 *
 * We spent sometimes with Mehdi Barkati for the ASTREE project which
 * deals with a similar issue for the FAUST language. It seems that
 * the late conversion is possible for the addition, the subtraction
 * and the multiplication because the modulo operation can be
 * distributed, in fact, factorized, but not for the division.
 *
 * For instance, -1 divided by 3 is equal to 0 or -1 depending on the
 * definition of the reminder, but 4294967295 divided by 3 cannot be
 * related to this result.
 *
 * This implies that unsigned arithmetic requires a full rework of
 * expression handling by the semantics analysis.
 */

#include <stdio.h>

int main(void)
{
  int i;
  unsigned int ui;

  i = -1;
  ui = i;

  fprintf(stderr, "ui = %u\n", ui);
}
