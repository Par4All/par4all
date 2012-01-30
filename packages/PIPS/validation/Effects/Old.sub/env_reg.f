C     the problem is to have coherent effects or regions at foo 
C     and bar call sites for variable /env/ a.
      program env_reg

C     regions and environment

      call foo

      call bar

      end

      subroutine foo
      common /env/ a(10), b(10)

      do i = 1, 10
         a(i) = 0.
      enddo

      end

      subroutine bar
      common /env/a(10), b(10)

      do i = 1, 10
         b(i) = a(i)
      enddo

      end

