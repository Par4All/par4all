! cloning on any constant...
      program c3
      integer i, n, m
      parameter (n=5, m = n + 1 - 2 + 1)
      i = 4
      i = i + 1
      call clonee(i)
      call clonee(n)
      call clonee(m)
      call clonee(5)
      call clonee(5 + n - m)
      end

      subroutine clonee(i)
      integer i
      if (i.eq.5) print *, 'ok'
      end
