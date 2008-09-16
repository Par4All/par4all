      program fs04
      integer i
      real s
      double precision d
      complex c
      complex*16 dc, dc2
      i = 145
      s = i+1
      d = s*5321.1212
      c = d+3.12
      dc = c + (1.00,2.00)
      i = d
      dc2 = c + i
      print *, dc, dc2
      end
