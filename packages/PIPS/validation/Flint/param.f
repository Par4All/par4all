      program param
      integer i, j, ir, id, ifl, io, il
      parameter (i = 12)
      parameter (ir = 12.0E0)
      parameter (id = 12.0D0)
      parameter (ifl = 12.0)
      parameter (io = i + 1)
      parameter (il = .TRUE.)
      real r, k, ri, rd, rf, ro, ror, rl
      parameter (r = 12.0E0)
      parameter (ri = 12)
      parameter (rd = 12.0D0)
      parameter (rf = 12.0)
      parameter (ro = i + 1.0)
      parameter (ror = r + 1.0E0)
      parameter (rl = .TRUE.)
      double precision d, di, df, dr, dop, doi, dl
      parameter (d = 12.0D0)
      parameter (di = 12)
      parameter (df = 12.0)
      parameter (dr = 12.E0)
      parameter (dop = d + 1.0)
      parameter (doi = d + 1)
      parameter (dl = .TRUE.)
      logical l, l1, l2, l3, l4, li, lr, ld
      parameter (l = .TRUE.)
      parameter (l1 = .FALSE.)
      parameter (l2 = l1.OR.l)
      parameter (l3 = 1)
      parameter (l4 = 1.AND..TRUE.)
      parameter (li = 1)
      parameter (lr = 1.0)
      parameter (ld = 1.0D0)
      print *, i, r, d
      j = i + 5
      k = i + 5.9
      print *, i, ir, id, ifl, io
      print *, r, ri, rd, rf, ro, ror
      print *, d, di, df, dr, dop, doi
      print *, l, l1, l2
! 8 errors...
      print *, il, dl, rl, li, lr, ld, l3, l4
      print *, il, dl, rl, li, lr, ld, l3, l4
      end
