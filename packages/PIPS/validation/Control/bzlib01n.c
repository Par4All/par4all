/* Attempt at simplifying bzlib.c */

#include "bzlib_private.h"

static Bool unRLE_obuf_to_output_FAST ( DState* s )
{
  UChar k1;

  if (s->blockRandomised) {

    while (True) {
      while (True) {
	if (s->strm->avail_out == 0) return False;
	if (s->state_out_len == 0) break;
	*( (UChar*)(s->strm->next_out) ) = s->state_out_ch;
      }

      /* can a new run be started? */
      if (s->nblock_used == s->save_nblock+1) return False;

      /* Only caused by corrupt data stream? */
      if (s->nblock_used > s->save_nblock+1)
	return True;

      s->state_out_len = 1;
      s->state_out_ch = s->k0;
      BZ_GET_FAST(k1); BZ_RAND_UPD_MASK;
      k1 ^= BZ_RAND_MASK; s->nblock_used++;
      if (s->nblock_used == s->save_nblock+1) continue;
      if (k1 != s->k0) { s->k0 = k1; continue; };

      s->state_out_len = 2;
      BZ_GET_FAST(k1); BZ_RAND_UPD_MASK;
      k1 ^= BZ_RAND_MASK; s->nblock_used++;
      if (s->nblock_used == s->save_nblock+1) continue;
      if (k1 != s->k0) { s->k0 = k1; continue; };

      s->state_out_len = 3;
      BZ_GET_FAST(k1); BZ_RAND_UPD_MASK;
      k1 ^= BZ_RAND_MASK; s->nblock_used++;
      if (s->nblock_used == s->save_nblock+1) continue;
      if (k1 != s->k0) { s->k0 = k1; continue; };

      BZ_GET_FAST(k1); BZ_RAND_UPD_MASK;
      k1 ^= BZ_RAND_MASK; s->nblock_used++;
      s->state_out_len = ((Int32)k1) + 4;
      BZ_GET_FAST(s->k0); BZ_RAND_UPD_MASK;
      s->k0 ^= BZ_RAND_MASK; s->nblock_used++;
    }

  } else {

    /* restore */
    UInt32        c_calculatedBlockCRC = s->calculatedBlockCRC;
    UChar         c_state_out_ch       = s->state_out_ch;
    Int32         c_state_out_len      = s->state_out_len;
    Int32         c_nblock_used        = s->nblock_used;
    Int32         c_k0                 = s->k0;
    UInt32*       c_tt                 = s->tt;
    UInt32        c_tPos               = s->tPos;
    char*         cs_next_out          = s->strm->next_out;
    unsigned int  cs_avail_out         = s->strm->avail_out;
    /* end restore */

    UInt32       avail_out_INIT = cs_avail_out;
    Int32        s_save_nblockPP = s->save_nblock+1;
    unsigned int total_out_lo32_old;

    while (True) {

      /* try to finish existing run */
      if (c_state_out_len > 0) {
	while (True) {
	  if (cs_avail_out == 0) goto return_notr;
	  if (c_state_out_len == 1) break;
	  *( (UChar*)(cs_next_out) ) = c_state_out_ch;
	  BZ_UPDATE_CRC ( c_calculatedBlockCRC, c_state_out_ch );
	  c_state_out_len--;
	  cs_next_out++;
	  cs_avail_out--;
	}
      s_state_out_len_eq_one:
	{
	  if (cs_avail_out == 0) {
	    c_state_out_len = 1; goto return_notr;
	  };
	  *( (UChar*)(cs_next_out) ) = c_state_out_ch;
	  BZ_UPDATE_CRC ( c_calculatedBlockCRC, c_state_out_ch );
	  cs_next_out++;
	  cs_avail_out--;
	}
      }
      /* Only caused by corrupt data stream? */
      if (c_nblock_used > s_save_nblockPP)
	return True;

      /* can a new run be started? */
      if (c_nblock_used == s_save_nblockPP) {
	c_state_out_len = 0; goto return_notr;
      };
      c_state_out_ch = c_k0;
      BZ_GET_FAST_C(k1); c_nblock_used++;
      if (k1 != c_k0) {
	c_k0 = k1; goto s_state_out_len_eq_one;
      };
      if (c_nblock_used == s_save_nblockPP)
	goto s_state_out_len_eq_one;

      c_state_out_len = 2;
      BZ_GET_FAST_C(k1); c_nblock_used++;
      if (c_nblock_used == s_save_nblockPP) continue;
      if (k1 != c_k0) { c_k0 = k1; continue; };

      c_state_out_len = 3;
      BZ_GET_FAST_C(k1); c_nblock_used++;
      if (c_nblock_used == s_save_nblockPP) continue;
      if (k1 != c_k0) { c_k0 = k1; continue; };

      BZ_GET_FAST_C(k1); c_nblock_used++;
      c_state_out_len = ((Int32)k1) + 4;
      BZ_GET_FAST_C(c_k0); c_nblock_used++;
    }

  return_notr:
    total_out_lo32_old = s->strm->total_out_lo32;
    s->strm->total_out_lo32 += (avail_out_INIT - cs_avail_out);
    if (s->strm->total_out_lo32 < total_out_lo32_old)
      s->strm->total_out_hi32++;

    /* save */
    s->calculatedBlockCRC = c_calculatedBlockCRC;
    s->state_out_ch       = c_state_out_ch;
    s->state_out_len      = c_state_out_len;
    s->nblock_used        = c_nblock_used;
    s->k0                 = c_k0;
    s->tt                 = c_tt;
    s->tPos               = c_tPos;
    s->strm->next_out     = cs_next_out;
    s->strm->avail_out    = cs_avail_out;
    /* end save */
  }
  return False;
}
