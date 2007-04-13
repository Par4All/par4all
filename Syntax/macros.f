! trying macros...
      program macro
      integer j
      external bla
      integer bla

! let us deal with macros...
      succ(i) = i + 1
      
      j=0

! hey, here it is used...      
      print *, succ(j)
      print *, succ(bla(3))

! another macro
      foo(i, j) = (i + j - 2)

      print *, foo(2, 3)
      print *, foo(bla(1), 3)
      print *, foo(1+bla(4), 5)

! a third one
      next(i, j) = (i*i*i-j*j*j)

      print *, next(j,j)
      print *, next(bla(j),bla(2))
      print *, next(bla(k),bla(2))

      print *, next(succ(i),foo(1,2))

      end

      integer function bla(i)
      integer i, count
      data count /0/
      count = count + 1
      bla = count
      end
