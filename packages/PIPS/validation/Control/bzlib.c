

/*-------------------------------------------------------------*/
/*--- Library top-level functions.                          ---*/
/*---                                               bzlib.c ---*/
/*-------------------------------------------------------------*/

/*--
  This file is a part of bzip2 and/or libbzip2, a program and
  library for lossless, block-sorting data compression.

  Copyright (C) 1996-2005 Julian R Seward.  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. The origin of this software must not be misrepresented; you must 
     not claim that you wrote the original software.  If you use this 
     software in a product, an acknowledgment in the product 
     documentation would be appreciated but is not required.

  3. Altered source versions must be plainly marked as such, and must
     not be misrepresented as being the original software.

  4. The name of the author may not be used to endorse or promote 
     products derived from this software without specific prior written 
     permission.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  Julian Seward, Cambridge, UK.
  jseward@bzip.org
  bzip2/libbzip2 version 1.0 of 21 March 2000

  This program is based on (at least) the work of:
     Mike Burrows
     David Wheeler
     Peter Fenwick
     Alistair Moffat
     Radford Neal
     Ian H. Witten
     Robert Sedgewick
     Jon L. Bentley

  For more information on these sources, see the manual.
--*/


#include "bzlib_private.h"


/*---------------------------------------------------*/
/* Return  True iff data corruption is discovered.
   Returns False if there is no problem.
*/
static
Bool unRLE_obuf_to_output_FAST ( DState* s )
{
  UChar k1;

  if (s->blockRandomised) {

    while (True) {
      /* try to finish existing run */
      while (True) {
	if (s->strm->avail_out == 0) return False;
	if (s->state_out_len == 0) break;
	*( (UChar*)(s->strm->next_out) ) = s->state_out_ch;
	BZ_UPDATE_CRC ( s->calculatedBlockCRC, s->state_out_ch );
	s->state_out_len--;
	s->strm->next_out++;
	s->strm->avail_out--;
	s->strm->total_out_lo32++;
	if (s->strm->total_out_lo32 == 0) s->strm->total_out_hi32++;
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
