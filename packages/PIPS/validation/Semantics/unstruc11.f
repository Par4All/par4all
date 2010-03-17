      program unstruc11

C     Check a scc which is path coverable but not a cycle

      k = 0
      i = 0
      j = 0
      l = 0

 100  continue
      if(x.gt.0.) go to 300
      l = l + 4
 200  continue
      k = k + 3
      i = i + 1
      if(y.gt.0.) go to 100
      j = j + 2
      go to 200

 300  continue
      print *, i, j, k, l

      end
