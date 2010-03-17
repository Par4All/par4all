      program while05

C     Do we identify some WHILE control structures?
C     I do not understand why this is recognized as a WHILE!!!
C     I should dump the unstructured...

      do i = 1, n
         call foo(i)
         if(x.lt.eps) go to 200
         call bar(i)
      enddo

      x = 1.

 200  continue

      end

      subroutine foo(i)
      end

      subroutine bar(i)
      end

