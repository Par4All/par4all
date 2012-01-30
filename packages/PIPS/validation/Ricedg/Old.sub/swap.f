      program swap
      integer i, j, t
      read *, i, j, t
      call swapfoo(i, i, t)
      print *, i, j, t
      read *, i, j, t
      call swapfoo(i, j, i)
      print *, i, j, t
      read *, i, j, t
      call swapfoo(i, j, j)
      print *, i, j, t
      end
      subroutine swapfoo(i, j, t)
      integer i, j, t
      t = i
      i = j
      j = t
      end
