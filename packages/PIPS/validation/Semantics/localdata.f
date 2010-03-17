      program localdata
      call foo
      call bla
      call foo
      end

      subroutine foo
! seems that m is implicitely saved, if not declared as so.
      integer m
      data m /5/
      save m
      print *, m
      m = m+1
      end

      subroutine bla
      integer n
      n = 10
      end
