! prettyprint of equivalence after cleaning declarations.
      program ee
      integer i1,i2,i3,i4,i5,i(5)
      common /foo/ i1,i2,i3,i4,i5
      equivalence (i(1),i1), (i(5),i5)
      read *, i2, i4
      print *, i2
      end
