      subroutine simd_constant_load2(vec, high, low)
      integer*4 vec(2)
      integer*4 high
      integer*4 low

      vec(0) = low
      vec(1) = high
      end

      subroutine simd_load2(vec, base, index, offset)
      integer*4 vec(2)
      integer*4 base(*)
      integer*4 index
      integer*4 offset
      
      vec(0) = base(index+offset)
      vec(1) = base(index+offset+1)
      end
      
      subroutine simd_generic_load2(vec, x1, x2)
      integer*4 vec(2)
      integer*4 x1
      integer*4 x2

      vec(0) = x1
      vec(1) = x2
      end

      subroutine simd_save2(vec, base, index, offset)
      integer*4 vec(2)
      integer*4 base(*)
      integer*4 index
      integer*4 offset
      
      base(index+offset) = vec(0)
      base(index+offset+1) = vec(1)
      end
      
      subroutine simd_generic_save2(vec, x1, x2)
      integer*4 vec(2)
      integer*4 x1
      integer*4 x2

      x1 = vec(0)
      x2 = vec(1)
      end

      subroutine simd_add2(dest, src1, src2)
      integer*4 dest(2)
      integer*4 src1(2)
      integer*4 src2(2)

      dest(0) = src1(0) + src2(0)
      dest(1) = src1(1) + src2(1)
      end

      subroutine simd_mul2(dest, src1, src2)
      integer*4 dest(2)
      integer*4 src1(2)
      integer*4 src2(2)

      dest(0) = src1(0) * src2(0)
      dest(1) = src1(1) * src2(1)
      end
      
      subroutine simd_constant_load4(vec, high, low)
      integer*2 vec(4)
      integer*4 high
      integer*4 low

      vec(0) = low AND 65535
      vec(1) = low / 65535
      vec(2) = high AND 65535
      vec(3) = high / 65535
      end

      subroutine simd_load4(vec, base, index, offset)
      integer*2 vec(4)
      integer*2 base(*)
      integer*4 index
      integer*4 offset
      
      vec(0) = base(index+offset)
      vec(1) = base(index+offset+1)
      vec(2) = base(index+offset+2)
      vec(3) = base(index+offset+3)
      end
      
      subroutine simd_generic_load4(vec, x1, x2, x3, x4)
      integer*2 vec(4)
      integer*2 x1
      integer*2 x2
      integer*2 x3
      integer*2 x4

      vec(0) = x1
      vec(1) = x2
      vec(2) = x3
      vec(3) = x4
      end

      subroutine simd_save4(vec, base, index, offset)
      integer*2 vec(4)
      integer*2 base(*)
      integer*4 index
      integer*4 offset
      
      base(index+offset) = vec(0)
      base(index+offset+1) = vec(1)
      base(index+offset+2) = vec(2)
      base(index+offset+3) = vec(3)
      end
      
      subroutine simd_generic_save4(vec, x1, x2, x3, x4)
      integer*2 vec(4)
      integer*2 x1
      integer*2 x2
      integer*2 x3
      integer*2 x4

      x1 = vec(0)
      x2 = vec(1)
      x3 = vec(2)
      x4 = vec(3)
      end

      subroutine simd_add4(dest, src1, src2)
      integer*2 dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(0) = src1(0) + src2(0)
      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      dest(3) = src1(3) + src2(3)
      end

      subroutine simd_mul4(dest, src1, src2)
      integer*2 dest(4)
      integer*2 src1(4)
      integer*2 src2(4)

      dest(0) = src1(0) * src2(0)
      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      dest(3) = src1(3) * src2(3)
      end

      subroutine simd_constant_load8(vec, high, low)
      integer*1 vec(8)
      integer*4 high
      integer*4 low

      vec(0) = low AND 255
      vec(1) = (low / 255) AND 255
      vec(2) = (low / 65535) AND 255
      vec(3) = (low / 16777216) AND 255
      vec(4) = high AND 255
      vec(5) = (high / 255) AND 255
      vec(6) = (high / 65535) AND 255
      vec(7) = (high / 16777216) AND 255
      end

      subroutine simd_load8(vec, base, index, offset)
      integer*1 vec(8)
      integer*1 base(*)
      integer*4 index
      integer*4 offset
      
      vec(0) = base(index+offset)
      vec(1) = base(index+offset+1)
      vec(2) = base(index+offset+2)
      vec(3) = base(index+offset+3)
      vec(4) = base(index+offset+4)
      vec(5) = base(index+offset+5)
      vec(6) = base(index+offset+6)
      vec(7) = base(index+offset+7)
      end
      
      subroutine simd_generic_load8(vec, x1, x2, x3, x4, x5, x6, x7, x8)
      integer*1 vec(8)
      integer*1 x1
      integer*1 x2
      integer*1 x3
      integer*1 x4
      integer*1 x5
      integer*1 x6
      integer*1 x7
      integer*1 x8

      vec(0) = x1
      vec(1) = x2
      vec(2) = x3
      vec(3) = x4
      vec(4) = x5
      vec(5) = x6
      vec(6) = x7
      vec(7) = x8
      end

      subroutine simd_save8(vec, base, index, offset)
      integer*1 vec(8)
      integer*1 base(*)
      integer*4 index
      integer*4 offset
      
      base(index+offset) = vec(0)
      base(index+offset+1) = vec(1)
      base(index+offset+2) = vec(2)
      base(index+offset+3) = vec(3)
      base(index+offset+4) = vec(4)
      base(index+offset+5) = vec(5)
      base(index+offset+6) = vec(6)
      base(index+offset+7) = vec(7)
      end
      
      subroutine simd_generic_save8(vec, x1, x2, x3, x4, x5, x6, x7, x8)
      integer*1 vec(8)
      integer*1 x1
      integer*1 x2
      integer*1 x3
      integer*1 x4
      integer*1 x5
      integer*1 x6
      integer*1 x7
      integer*1 x8

      x1 = vec(0)
      x2 = vec(1)
      x3 = vec(2)
      x4 = vec(3)
      x5 = vec(4)
      x6 = vec(5)
      x7 = vec(6)
      x8 = vec(7)
      end

      subroutine simd_add8(dest, src1, src2)
      integer*1 dest(8)
      integer*1 src1(8)
      integer*1 src2(8)

      dest(0) = src1(0) + src2(0)
      dest(1) = src1(1) + src2(1)
      dest(2) = src1(2) + src2(2)
      dest(3) = src1(3) + src2(3)
      dest(4) = src1(4) + src2(4)
      dest(5) = src1(5) + src2(5)
      dest(6) = src1(6) + src2(6)
      dest(7) = src1(7) + src2(7)
      end

      subroutine simd_mul8(dest, src1, src2)
      integer*1 dest(8)
      integer*1 src1(8)
      integer*1 src2(8)

      dest(0) = src1(0) * src2(0)
      dest(1) = src1(1) * src2(1)
      dest(2) = src1(2) * src2(2)
      dest(3) = src1(3) * src2(3)
      dest(4) = src1(4) * src2(4)
      dest(5) = src1(5) * src2(5)
      dest(6) = src1(6) * src2(6)
      dest(7) = src1(7) * src2(7)
      end

      subroutine simd_mov2(dest, src)
      integer*4 dest(2)
      integer*4 src(2)

      dest(0) = src(0)
      dest(1) = src(1)
      end

      subroutine simd_mov4(dest, src)
      integer*2 dest(4)
      integer*2 src(4)

      dest(0) = src(0)
      dest(1) = src(1)
      dest(2) = src(2)
      dest(3) = src(3)
      end

      subroutine simd_mov8(dest, src)
      integer*1 dest(8)
      integer*1 src(8)

      dest(0) = src(0)
      dest(1) = src(1)
      dest(2) = src(2)
      dest(3) = src(3)
      dest(4) = src(4)
      dest(5) = src(5)
      dest(6) = src(6)
      dest(7) = src(7)
      end

      subroutine simd_opp2(dest, src)
      integer*4 dest(2)
      integer*4 src(2)

      dest(0) = -src(0)
      dest(1) = -src(1)
      end

      subroutine simd_opp4(dest, src)
      integer*2 dest(4)
      integer*2 src(4)

      dest(0) = -src(0)
      dest(1) = -src(1)
      dest(2) = -src(2)
      dest(3) = -src(3)
      end

      subroutine simd_opp8(dest, src)
      integer*1 dest(8)
      integer*1 src(8)

      dest(0) = -src(0)
      dest(1) = -src(1)
      dest(2) = -src(2)
      dest(3) = -src(3)
      dest(4) = -src(4)
      dest(5) = -src(5)
      dest(6) = -src(6)
      dest(7) = -src(7)
      end
