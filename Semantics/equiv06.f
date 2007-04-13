      program equiv06

C     Check that triplets of equivalenced variables are properly processed

      equivalence (i1, i2)
      equivalence (i1, i3)

      equivalence (j3, j2)
      equivalence (j1, j3)

      equivalence (k3, k2)
      equivalence (k2, k1)

      read *, i1
      read *, i2
      read *, i3

      read *, j3
      read *, j2
      read *, j1

      read *, k2
      read *, k3
      read *, k1

      print *, i3, j2, k1

      end
