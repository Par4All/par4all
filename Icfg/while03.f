      program while03

C     Do we identify some WHILE control structures?

      do i = 1, n
         call foo(i)
         if(x.lt.eps) go to 200
      enddo

 200  continue

      end

      subroutine foo
      end

