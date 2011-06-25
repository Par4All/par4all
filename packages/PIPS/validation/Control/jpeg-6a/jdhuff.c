/* Derived from static function mcu_decode in jpeg-6a
 *
 * The for loop is necessary to show the bug
 */

int decode_mcu ()
{
  int blkn;

  for (blkn = 0; blkn < 64; blkn++) {
    if (! blkn)
      goto skip_ACs;
    if (blkn > 1) {
      ;
    } else {
    skip_ACs:
      ;
    }
  }
  return 1;
}
