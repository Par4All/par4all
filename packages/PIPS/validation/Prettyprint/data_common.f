      program dc
! here is common foo
      integer i, j
      common /foo/ i, j
      print *, i, j
      call bla
      end

      subroutine bla
! here is common foo and associated data
! it must be kept because of the data...
      integer i, j
      common /foo/ i, j
      data i, j / 1, 2 /
      print *, "hello world"
      end
