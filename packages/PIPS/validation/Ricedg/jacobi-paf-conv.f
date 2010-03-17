      program jacobi
      integer count, maxcyc, p, q, i
      real a(100, 100), x
      real app, apq, aqq, theta, c, s, csq, ssq, cs, aip, aiq
      do 10 count = 1, maxcyc
         do 10 p=1,n-1
            do 10 q = p+1,n
               app = a(p,p)
               apq = a(p,q)
               aqq = a(q,q)
               theta = atan2(2.0*apq, app-aqq)/2.0
               c = cos(theta)
               s = sin(theta)
               csq = c*c
               ssq = s*s
               cs = c*s
               a(p,p) = csq * app + 2.0*cs*apq + ssq * aqq
               a(q,q) = ssq * app - 2.0*cs*apq + csq * aqq
               a(p,q) = 0.0
               a(q,p) = 0.0
               do 20 i=1, p-1                  
                  aip = a(i,p)
                  aiq = a(i,q)
                  a(i,p) = c*aip + s*aiq
                  a(p,i) = a(i,p)
                  a(i,q) = -s*aip + c*aiq
                  a(q,i) = a(i,q)             
 20            continue
               do 30 i=p+1,q-1
                  aip = a(i,p)
                  aiq = a(i,q)
                  a(i,p) = c*aip + s*aiq
                  a(p,i) = a(i,p)
                  a(i,q) = -s*aip + c*aiq
                  a(q,i) = a(i,q)
 30            continue
               do 40 i=q+1,n
                  aip = a(i,p)
                  aiq = a(i,q)
                  a(i,p) = c*aip + s*aiq
                  a(p,i) = a(i,p)
                  a(i,q) = -s*aip + c*aiq
                  a(q,i) = a(i,q)
 40            continue
 10         continue
      end
