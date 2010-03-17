      integer function simd_phi(l, x1, x2)
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

      subroutine simd_load_v4sf(vec, base)
      real*4 vec(4)
      real*4 base(4)
      real*4 index
      real*4 offset
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
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

      subroutine simd_save_v4sf(vec, base)
      real*4 vec(4)
      real*4 base(4)
      real*4 index
      real*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      end
      
      subroutine simd_save_generic_v4sf(vec, x1, x2, x3, x4)
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

      subroutine simd_subps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) - src2(1)
      dest(2) = src1(2) - src2(2)
      dest(3) = src1(3) - src2(3)
      dest(4) = src1(4) - src2(4)
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

      subroutine simd_divps(dest, src1, src2)
      real*4 dest(4)
      real*4 src1(4)
      real*4 src2(4)

      dest(1) = src1(1) / src2(1)
      dest(2) = src1(2) / src2(2)
      dest(3) = src1(3) / src2(3)
      dest(4) = src1(4) / src2(4)
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

      subroutine simd_save_v2si(vec, base)
      integer*4 vec(2)
      integer*4 base(2)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      end
      
      subroutine simd_save_generic_v2si(vec, x1, x2)
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

      subroutine simd_save_v4si(vec, base)
      integer*4 vec(4)
      integer*4 base(4)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      end
      
      subroutine simd_save_generic_v4si(vec, x1, x2, x3, x4)
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
      
      subroutine simd_load_constant_v4hi(vec, x1, x2, x3, x4)
      integer*2 vec(4)
      integer*2 x1, x2, x3, x4

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      end

      subroutine simd_load_v4hi(vec, base)
      integer*2 vec(4)
      integer*2 base(4)
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
      end

      subroutine simd_load_v4qi_to_v4hi(vec, base)
      integer*2 vec(4)
      integer*1 base(4)
      
      vec(1) = base(1)
      vec(2) = base(2)
      vec(3) = base(3)
      vec(4) = base(4)
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

      subroutine simd_save_v4hi(vec, base)
      integer*2 vec(4)
      integer*2 base(4)
      integer*4 index
      integer*4 offset
      
      base(1) = vec(1)
      base(2) = vec(2)
      base(3) = vec(3)
      base(4) = vec(4)
      end
      
      subroutine simd_save_generic_v4hi(vec, x1, x2, x3, x4)
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

      subroutine simd_mulw(dest, src1, src2)
      integer*2 dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      dest(3) = src1(3) * src2(3)
      dest(4) = src1(4) * src2(4)
      end

      subroutine simd_load_constant_v8qi(vec, x1, x2, x3, x4,  x5, x6, 
     &x7, x8)
      integer*1 vec(8)
      integer*4 high
      integer*4 low

      integer*4 temp

      vec(1) = x1
      vec(2) = x2
      vec(3) = x3
      vec(4) = x4
      vec(5) = x5
      vec(6) = x6
      vec(7) = x7
      vec(8) = x8
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

      subroutine simd_save_v8qi(vec, base)
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
      
      subroutine simd_save_generic_v8qi(vec, x1, x2, x3, x4, x5, x6, x7,
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
