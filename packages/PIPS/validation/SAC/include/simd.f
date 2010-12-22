      integer function phi(l, x1, x2)
      implicit none
      logical l
      integer x, x1, x2

      if(L) then
         x = x1
      else
         x = x2
      endif

      phi = x
      end
      subroutine SIMD_STORE_V2DF_TO_V2SF(vdc,vdi)
      real*8 vdi(2)
      real*4 vdc(2)
      vdc(1)=vdi(1)
      vdc(2)=vdi(2)
      end
      subroutine SIMD_LOAD_V2DF_TO_V2SF(vdc,vdi)
      real*8 vdi(2)
      real*4 vdc(2)
      vdc(1)=vdi(1)
      vdc(2)=vdi(2)
      end
      subroutine SIMD_LOAD_V2DC_TO_V2DI(vdc,vdi)
      integer*8 vdi(2)
      complex*8 vdc(2)
      vdc(1)=vdi(1)
      vdc(2)=vdi(2)
      end
      subroutine SIMD_STORE_V2DC_TO_V2DI(vdc,vdi)
      integer*8 vdi(2)
      complex*8 vdc(2)
      vdc(1)=vdi(1)
      vdc(2)=vdi(2)
      end

      subroutine SIMD_LOAD_V2SF_TO_V2DF(vsf,vsi)
      real*4 vsf(2)
      real*8 vsi(2)
      vsf(1)=vsi(1)
      vsf(2)=vsi(2)
      end

      subroutine SIMD_STORE_V2SF_TO_V2DF(vdf,vsf)
      real*4 vsf(2)
      real*8 vdf(2)
      vdf(1)=vsf(1)
      vdf(2)=vsf(2)
      end

      subroutine SIMD_LOAD_V4SI_TO_V4SF(vsf,vsi)
      real*4 vsf(4)
      integer*4 vsi(4)
      vsf(1)=vsi(1)
      vsf(2)=vsi(2)
      vsf(3)=vsi(3)
      vsf(4)=vsi(4)
      end

      subroutine simd_load_v4sf(vec, base)
      real*4 vec(4)
      real*4 base(4)
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
      end
      subroutine simd_load_v2di(vec, base)
      integer*8 vec(2)
      integer*8 base(2)
      
      vec(1) = base(1)
      vec(2) = base(2)
      end
      subroutine simd_load_v2df(vec, base)
      real*8 vec(2)
      real*8 base(2)
      
      vec(1) = base(1)
      vec(2) = base(2)
      end
      
      subroutine simd_load_generic_v4sf(vec, x1, x2, x3, x4)
      real*4 vec(4)
      real*4 x1
      real*4 x2
      real*4 x3
      real*4 x4

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      end
      subroutine simd_load_generic_v2di(vec, x1, x2)
      integer*8 vec(2)
      integer*8 x1
      integer*8 x2

      vec(1) = x1
      vec(2) = x2
      end
      subroutine simd_load_generic_v2df(vec, x1, x2)
      real*8 vec(2)
      real*8 x1
      real*8 x2

      vec(1) = x1
      vec(2) = x2
      end

      subroutine simd_load_constant_v4sf(vec, x1, x2, x3, x4)
      real*4 vec(4)
      real*4 x1
      real*4 x2
      real*4 x3
      real*4 x4

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      end
      subroutine simd_load_constant_v2df(vec, x1, x2 )
      real*8 vec(2)
      real*8 x1
      real*8 x2

      vec(1) = x1
      vec(2) = x2
      end

      subroutine simd_store_v4sf(vec, base)
      real*4 vec(4)
      real*4 base(4)
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      end
      subroutine simd_store_v2di(vec, base)
      integer*8 vec(2)
      integer*8 base(2)
      
      base(1) = vec(1)
      base(2) = vec(2)
      end
      subroutine simd_store_v2df(vec, base)
      real*8 vec(2)
      real*8 base(2)
      
      base(1) = vec(1)
      base(2) = vec(2)
      end
      
      subroutine simd_store_generic_v4sf(vec, x1, x2, x3, x4)
      real*4 vec(4)
      real*4 x1
      real*4 x2
      real*4 x3
      real*4 x4

      x1 = vec(1)
      x2 = vec(2)
      x3 = vec(3)
      x4 = vec(4)
      end
      subroutine simd_store_generic_v2df(vec, x1, x2)
      real*8 vec(2)
      real*8 x1
      real*8 x2

      x1 = vec(1)
      x2 = vec(2)
      end

      subroutine simd_cmpgtps(dest, src1, src2)
      logical dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) .GT. src2(1)
      dest(2) = src1(2) .GT. src2(2)
      dest(3) = src1(3) .GT. src2(3)
      dest(4) = src1(4) .GT. src2(4)
      end

      subroutine simd_phips(dest, cond, src1, src2)
      real*4 dest(4)
      logical cond(4)
      real*4 src1(4)
      real*4 src2(4)

      if(cond(1)) then
         dest(1) = src1(1)
      else
         dest(1) = src2(1)
      endif

      if(cond(2)) then
         dest(2) = src1(2)
      else
         dest(2) = src2(2)
      endif

      if(cond(3)) then
         dest(3) = src1(3)
      else
         dest(3) = src2(3)
      endif

      if(cond(4)) then
         dest(4) = src1(4)
      else
         dest(4) = src2(4)
      endif
      end

      subroutine simd_addps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      dest(3) = src1(3) + src2(3)
      dest(4) = src1(4) + src2(4)
      end
      subroutine simd_addpd(dest, src1, src2)
      real*8 dest(2)
      real*8 src1(2)
      real*8 src2(2)

      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      end
      subroutine simd_uminps(dest, src1)
      real*4 dest(4)
      real*4 src1(4)

      dest(1) =  - src1(1)
      dest(2) =  - src1(2)
      dest(3) =  - src1(3)
      dest(4) =  - src1(4)
      end

      subroutine simd_subps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) - src2(1)
      dest(2) = src1(2) - src2(2)
      dest(3) = src1(3) - src2(3)
      dest(4) = src1(4) - src2(4)
      end

      subroutine simd_addcs(dest, src1, src2)
      complex dest(2)
      complex src1(2)
      complex src2(2)

      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      end

      subroutine simd_mulcs(dest, src1, src2)
      complex dest(2)
      complex src1(2)
      complex src2(2)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      end

      subroutine simd_subpd(dest, src1, src2)
      real*8 dest(2)
      real*8 src1(2)
      real*8 src2(2)

      dest(1) = src1(1) - src2(1)
      dest(2) = src1(2) - src2(2)
      end

      subroutine simd_mulps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      dest(3) = src1(3) * src2(3)
      dest(4) = src1(4) * src2(4)
      end
      subroutine simd_mulpd(dest, src1, src2)
      real*8 dest(2)
      real*8 src1(2)
      real*8 src2(2)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      end

      subroutine simd_divps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) / src2(1)
      dest(2) = src1(2) / src2(2)
      dest(3) = src1(3) / src2(3)
      dest(4) = src1(4) / src2(4)
      end

      subroutine simd_divpd(dest, src1, src2)
      real*8 dest(2)
      real*8 src1(2)
      real*8 src2(2)

      dest(1) = src1(1) / src2(1)
      dest(2) = src1(2) / src2(2)
      end

      subroutine simd_maxps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = MAX(src1(1), src2(1))
      dest(2) = MAX(src1(2), src2(2))
      dest(3) = MAX(src1(3), src2(3))
      dest(4) = MAX(src1(4), src2(4))
      end

      subroutine simd_load_constant_v2si(vec, high, low)
      integer*4 vec(2)
      integer*4 high
      integer*4 low

      vec(1) = low
      vec(2) = high
      end

      subroutine simd_load_v2si(vec, base)
      integer*4 vec(2)
      integer*4 base(2)
      integer*4 index
      integer*4 offset
      
      vec(1) = base(1)
      vec(2) = base(2)
      end
      
      subroutine simd_load_generic_v2si(vec, x1, x2)
      integer*4 vec(2)
      integer*4 x1
      integer*4 x2

      vec(1) = x1
      vec(2) = x2
      end

      subroutine simd_store_v2si(vec, base)
      integer*4 vec(2)
      integer*4 base(2)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      end
      subroutine simd_store_generic_v2di(vec, x1, x2)
      integer*8 vec(2)
      integer*8 x1
      integer*8 x2

      x1 = vec(1)
      x2 = vec(2)
      end
      
      subroutine simd_store_generic_v2si(vec, x1, x2)
      integer*4 vec(2)
      integer*4 x1
      integer*4 x2

      x1 = vec(1)
      x2 = vec(2)
      end

      subroutine simd_load_v4si(vec, base)
      integer*4 vec(4)
      integer*4 base(4)
      integer*4 index
      integer*4 offset
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
      end
      
      subroutine simd_load_generic_v4si(vec, x1, x2, x3, x4)
      integer*4 vec(4)
      integer*4 x1
      integer*4 x2
      integer*4 x3
      integer*4 x4

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      end

      subroutine simd_store_v4si(vec, base)
      integer*4 vec(4)
      integer*4 base(4)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      end
      
      subroutine simd_store_generic_v4si(vec, x1, x2, x3, x4)
      integer*4 vec(4)
      integer*4 x1
      integer*4 x2
      integer*4 x3
      integer*4 x4

      x1 = vec(1)
      x2 = vec(2)
      x3 = vec(3)
      x4 = vec(4)
      end

      subroutine simd_addd(dest, src1, src2)
      integer*4 dest(2)
      integer*4 src1(2)
      integer*4 src2(2)

      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      end

      subroutine simd_subd(dest, src1, src2)
      integer*4 dest(2)
      integer*4 src1(2)
      integer*4 src2(2)

      dest(1) = src1(1) - src2(1)
      dest(2) = src1(2) - src2(2)
      end

      subroutine simd_muld(dest, src1, src2)
      integer*4 dest(2)
      integer*4 src1(2)
      integer*4 src2(2)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      end
      
      subroutine simd_load_constant_v4hi(vec, high, low)
      integer*2 vec(4)
      integer*4 high
      integer*4 low

      vec(1) = low AND 65535
      vec(2) = low / 65535
      vec(3) = high AND 65535
      vec(4) = high / 65535
      end

      subroutine simd_load_v4hi(vec, base)
      integer*2 vec(4)
      integer*2 base(4)
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
      end
      subroutine simd_load_v8hi(vec, base)
      integer*1 vec(8)
      integer*1 base(8)
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
      vec(5) = base(5)
      vec(6) = base(6)
      vec(7) = base(7)
      vec(8) = base(8)
      end
      subroutine simd_load_generic_v8hi(v,b0,b1,b2,b3,b4,b5,b6,b7)
      integer*1 v(8)
      integer*1 b0
      integer*1 b1
      integer*1 b2
      integer*1 b3
      integer*1 b4
      integer*1 b5
      integer*1 b6
      integer*1 b7
      
      v(1) = b1
      v(2) = b2
      v(3) = b3
      v(4) = b4
      v(5) = b5
      v(6) = b6
      v(7) = b7
      v(8) = b8
      end

      subroutine simd_load_generic_v4hi(vec, x1, x2, x3, x4)
      integer*2 vec(4)
      integer*2 x1
      integer*2 x2
      integer*2 x3
      integer*2 x4

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      end

      subroutine simd_store_v4hi(vec, base)
      integer*2 vec(4)
      integer*2 base(4)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      end
      
      subroutine simd_store_v8hi(vec, base)
      integer*1 vec(8)
      integer*1 base(8)
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      base(5) = vec(5)
      base(6) = vec(6)
      base(7) = vec(7)
      base(8) = vec(8)
      end

      subroutine simd_store_generic_v8hi(v, x1, x2, x3, x4,x5,x6,x7,x8)
      integer*2 v(8)
      integer*2 x1
      integer*2 x2
      integer*2 x3
      integer*2 x4
      integer*2 x5
      integer*2 x6
      integer*2 x7
      integer*2 x8

      x1 = v(1)
      x2 = v(2)
      x3 = v(3)
      x4 = v(4)
      x5 = v(5)
      x6 = v(6)
      x7 = v(7)
      x8 = v(8)
      end
      subroutine simd_store_generic_v4hi(vec, x1, x2, x3, x4)
      integer*2 vec(4)
      integer*2 x1
      integer*2 x2
      integer*2 x3
      integer*2 x4

      x1 = vec(1)
      x2 = vec(2)
      x3 = vec(3)
      x4 = vec(4)
      end

      subroutine simd_cmpgtw(dest, src1, src2)
      logical dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(1) = src1(1) .GT. src2(1)
      dest(2) = src1(2) .GT. src2(2)
      dest(3) = src1(3) .GT. src2(3)
      dest(4) = src1(4) .GT. src2(4)
      end

      subroutine simd_phiw(dest, cond, src1, src2)
      integer*2 dest(4)
      logical cond(4)
      integer*2 src1(4)
      integer*2 src2(4)

      if(cond(1)) then
         dest(1) = src1(1)
      else
         dest(1) = src2(1)
      endif

      if(cond(2)) then
         dest(2) = src1(2)
      else
         dest(2) = src2(2)
      endif

      if(cond(3)) then
         dest(3) = src1(3)
      else
         dest(3) = src2(3)
      endif

      if(cond(4)) then
         dest(4) = src1(4)
      else
         dest(4) = src2(4)
      endif
      end

      subroutine simd_addw(dest, src1, src2)
      integer*2 dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      dest(3) = src1(3) + src2(3)
      dest(4) = src1(4) + src2(4)
      end

      subroutine simd_subw(dest, src1, src2)
      integer*2 dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(1) = src1(1) - src2(1)
      dest(2) = src1(2) - src2(2)
      dest(3) = src1(3) - src2(3)
      dest(4) = src1(4) - src2(4)
      end

      subroutine simd_mulw(dest, src1, src2)
      integer*2 dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      dest(3) = src1(3) * src2(3)
      dest(4) = src1(4) * src2(4)
      end

      subroutine simd_load_constant_v8qi(vec, high, low)
      integer*1 vec(8)
      integer*4 high
      integer*4 low

      integer*4 temp

      vec(1) = low AND 255
      temp = (low / 256)
      vec(2) = temp AND 255
      temp = (low / 65536)
      vec(3) = temp AND 255
      temp = (low / 16777216)
      vec(4) = temp AND 255
      vec(5) = high AND 255
      temp = (high / 256)
      vec(6) = temp AND 255
      temp = (high / 65536)
      vec(7) = temp AND 255
      temp = (high / 16777216)
      vec(8) = temp AND 255
      end

      subroutine simd_load_v8qi(vec, base)
      integer*1 vec(8)
      integer*1 base(8)
      integer*4 index
      integer*4 offset
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
      vec(5) = base(5)
      vec(6) = base(6)
      vec(7) = base(7)
      vec(8) = base(8)
      end
      
      subroutine simd_load_generic_v8qi(vec, x1, x2, x3, x4, x5, x6, x7,
     & x8)
      integer*1 vec(8)
      integer*1 x1
      integer*1 x2
      integer*1 x3
      integer*1 x4
      integer*1 x5
      integer*1 x6
      integer*1 x7
      integer*1 x8

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      vec(5) = x5
      vec(6) = x6
      vec(7) = x7
      vec(8) = x8
      end

      subroutine simd_store_v8qi(vec, base)
      integer*1 vec(8)
      integer*1 base(*)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      base(5) = vec(5)
      base(6) = vec(6)
      base(7) = vec(7)
      base(8) = vec(8)
      end
      
      subroutine simd_store_generic_v8qi(vec, x1, x2, x3, x4, x5, x6, x7,
     & x8)
      integer*1 vec(8)
      integer*1 x1
      integer*1 x2
      integer*1 x3
      integer*1 x4
      integer*1 x5
      integer*1 x6
      integer*1 x7
      integer*1 x8

      x1 = vec(1)
      x2 = vec(2)
      x3 = vec(3)
      x4 = vec(4)
      x5 = vec(5)
      x6 = vec(6)
      x7 = vec(7)
      x8 = vec(8)
      end

      subroutine simd_addb(dest, src1, src2)
      integer*1 dest(8)
      integer*1 src1(8)
      integer*1 src2(8)

      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      dest(3) = src1(3) + src2(3)
      dest(4) = src1(4) + src2(4)
      dest(5) = src1(5) + src2(5)
      dest(6) = src1(6) + src2(6)
      dest(7) = src1(7) + src2(7)
      dest(8) = src1(8) + src2(8)
      end

      subroutine simd_subb(dest, src1, src2)
      integer*1 dest(8)
      integer*1 src1(8)
      integer*1 src2(8)

      dest(1) = src1(1) - src2(1)
      dest(2) = src1(2) - src2(2)
      dest(3) = src1(3) - src2(3)
      dest(4) = src1(4) - src2(4)
      dest(5) = src1(5) - src2(5)
      dest(6) = src1(6) - src2(6)
      dest(7) = src1(7) - src2(7)
      dest(8) = src1(8) - src2(8)
      end

      subroutine simd_mulb(dest, src1, src2)
      integer*1 dest(8)
      integer*1 src1(8)
      integer*1 src2(8)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      dest(3) = src1(3) * src2(3)
      dest(4) = src1(4) * src2(4)
      dest(5) = src1(5) * src2(5)
      dest(6) = src1(6) * src2(6)
      dest(7) = src1(7) * src2(7)
      dest(8) = src1(8) * src2(8)
      end

      subroutine simd_movps(dest, src)
      real*4 dest(2)
      real*4 src(2)

      dest(1) = src(1)
      dest(2) = src(2)
      end

      subroutine simd_movd(dest, src)
      integer*4 dest(2)
      integer*4 src(2)

      dest(1) = src(1)
      dest(2) = src(2)
      end

      subroutine simd_movw(dest, src)
      integer*2 dest(4)
      integer*2 src(4)

      dest(1) = src(1)
      dest(2) = src(2)
      dest(3) = src(3)
      dest(4) = src(4)
      end

      subroutine simd_movb(dest, src)
      integer*1 dest(8)
      integer*1 src(8)

      dest(1) = src(1)
      dest(2) = src(2)
      dest(3) = src(3)
      dest(4) = src(4)
      dest(5) = src(5)
      dest(6) = src(6)
      dest(7) = src(7)
      dest(8) = src(8)
      end

      subroutine simd_oppps(dest, src)
      real*4 dest(2)
      real*4 src(2)

      dest(1) = -src(1)
      dest(2) = -src(2)
      end

      subroutine simd_oppd(dest, src)
      integer*4 dest(2)
      integer*4 src(2)

      dest(1) = -src(1)
      dest(2) = -src(2)
      end

      subroutine simd_oppw(dest, src)
      integer*2 dest(4)
      integer*2 src(4)

      dest(1) = -src(1)
      dest(2) = -src(2)
      dest(3) = -src(3)
      dest(4) = -src(4)
      end

      subroutine simd_oppb(dest, src)
      integer*1 dest(8)
      integer*1 src(8)

      dest(1) = -src(1)
      dest(2) = -src(2)
      dest(3) = -src(3)
      dest(4) = -src(4)
      dest(5) = -src(5)
      dest(6) = -src(6)
      dest(7) = -src(7)
      dest(8) = -src(8)
      end

