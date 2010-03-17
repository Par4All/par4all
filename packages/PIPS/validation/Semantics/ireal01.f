      program ireal01

C     Bug found in spice.f from the PerfectClub Benchmark

C     i=MIN(MAX(k0, k1), k2) should not generate any constraint
C     but I <= k2

      real value(100)

      i = 1.2
      i = min(1.2, k)
      LWIDTH=MIN(MAX(IFIELD,72),132)
      LWIDTH=MIN(MAX(IFIELD,72.0D0),132)

      LWIDTH=MIN(MAX(IFIELD,72),132.0D0)

      LWIDTH=MIN(MAX(IFIELD,72.0D0),132.0D0)
      LWIDTH=MIN(MAX((IFIELD+IFLD),72.0D0),132.0D0)
      LWIDTH=MIN(MAX(VALUE(IFIELD+IFLD),72.0D0),132.0D0)

      print *, i

      end
