/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * xzlarfg.cpp
 *
 * Code generation for function 'xzlarfg'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "processIMU.h"
#include "xzlarfg.h"
#include "qrsolve.h"
#include "xnrm2.h"
#include "processIMU_rtwutil.h"

/* Function Definitions */
double xzlarfg(int n, double *alpha1, double x[4096], int ix0)
{
  double tau;
  double xnorm;
  int knt;
  int i10;
  int k;
  tau = 0.0;
  xnorm = b_xnrm2(n - 1, x, ix0);
  if (xnorm != 0.0) {
    xnorm = rt_hypotd_snf(*alpha1, xnorm);
    if (*alpha1 >= 0.0) {
      xnorm = -xnorm;
    }

    if (std::abs(xnorm) < 1.0020841800044864E-292) {
      knt = 0;
      i10 = (ix0 + n) - 2;
      do {
        knt++;
        for (k = ix0; k <= i10; k++) {
          x[k - 1] *= 9.9792015476736E+291;
        }

        xnorm *= 9.9792015476736E+291;
        *alpha1 *= 9.9792015476736E+291;
      } while (!(std::abs(xnorm) >= 1.0020841800044864E-292));

      xnorm = rt_hypotd_snf(*alpha1, b_xnrm2(n - 1, x, ix0));
      if (*alpha1 >= 0.0) {
        xnorm = -xnorm;
      }

      tau = (xnorm - *alpha1) / xnorm;
      *alpha1 = 1.0 / (*alpha1 - xnorm);
      i10 = (ix0 + n) - 2;
      for (k = ix0; k <= i10; k++) {
        x[k - 1] *= *alpha1;
      }

      for (k = 1; k <= knt; k++) {
        xnorm *= 1.0020841800044864E-292;
      }

      *alpha1 = xnorm;
    } else {
      tau = (xnorm - *alpha1) / xnorm;
      *alpha1 = 1.0 / (*alpha1 - xnorm);
      i10 = (ix0 + n) - 2;
      for (k = ix0; k <= i10; k++) {
        x[k - 1] *= *alpha1;
      }

      *alpha1 = xnorm;
    }
  }

  return tau;
}

/* End of code generation (xzlarfg.cpp) */
