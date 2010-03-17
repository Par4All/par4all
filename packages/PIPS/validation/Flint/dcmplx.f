      program dc
      double complex a, b, c, d, e, f, g, h
      double precision x
      complex w, z

      a = (1.0,1.0)
      b = 1
      c = 1.0e0
      d = 1.0d0

      e = a + b
      f = COS(e)
      x = ABS(f)
      f = DCMPLX(x)
      h = LOG(g) + SQRT(f) + SIN(e) + EXP(a)

      print *, h

      w = (1.0, 1.0)
      z = w + 1

      w = CMPLX(1.0, 1.0)
      z = w + 1

      end
