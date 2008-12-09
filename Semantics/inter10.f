      subroutine foo(i)
      end

      integer function inter10(i)
      inter10 =  i+1
      end

      program main
      integer i
      data i/4/
      i = inter10(i)
      call foo(i)
      end

