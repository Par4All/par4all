c      program env
c      subroutine env

C     Name conflicts between subroutines, functions, main and commons

      call foo

      call bar

      end

      subroutine foo
      common /env/ a(10), b(10)
      common /bar/ x, y

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
