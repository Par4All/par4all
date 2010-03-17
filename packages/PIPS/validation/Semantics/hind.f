      program hind

c     PLDI'95, Constant propagation

      call sub1(0)

      end

      subroutine sub1(f1)
      integer f1
      integer x, y
      
      x = 9

      if(f1.eq.0) then
         y = 1
      else
         y = 0
      endif

      call sub2(y, 4, f1, x)

      end

      subroutine sub2(f2, f3, f4, f5)
      integer f2, f3, f4, f5

      print *, f2+f3+f4+f5

      end
