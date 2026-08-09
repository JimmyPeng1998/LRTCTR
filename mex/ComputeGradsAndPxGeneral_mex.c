#include "mex.h"

/*
 * [Px, G1, ..., Gd] = ComputeGradsAndPxGeneral_mex(...)
 *
 * General-order sampled TR values and Euclidean gradients. Each output
 * gradient has the same r(k)-by-r(k+1)-by-n(k) layout as its input core.
 *
 * Reference: Quotient geometry of tensor ring decomposition,
 *    Bin Gao, Renfeng Peng, and Ya-xiang Yuan,
 *    arXiv preprint arXiv:2601.21874, 2026.
 *    https://arxiv.org/abs/2601.21874
 *
 * Original author: Renfeng Peng, Aug. 05, 2026.
 */

static void multiply(const double *A, const double *B, double *C,
                     mwSize rows, mwSize inner, mwSize columns)
{
    mwSize i, j, ell;
    for (j = 0; j < columns; ++j) {
        for (i = 0; i < rows; ++i) {
            double value = 0.0;
            for (ell = 0; ell < inner; ++ell) {
                value += A[i+ell*rows]*B[ell+j*inner];
            }
            C[i+j*rows] = value;
        }
    }
}

static void identity(double *A, mwSize size)
{
    mwSize i;
    for (i = 0; i < size*size; ++i) {
        A[i] = 0.0;
    }
    for (i = 0; i < size; ++i) {
        A[i+i*size] = 1.0;
    }
}

static void require_uint32(const mxArray *array, const char *name)
{
    if (!mxIsUint32(array) || mxIsComplex(array)) {
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:Type",
            "%s must be a real uint32 array.", name);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize d, sample_count, sample, k, i, j, ell;
    const uint32_T *n;
    const uint32_T *r;
    const uint32_T *Omega;
    const double **cores;
    const double *PA;
    double **gradients;
    double **prefix;
    double **suffix;
    double *values;
    double p;

    if (nrhs < 10) {
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:Inputs",
            "At least three tensor cores are required.");
    }
    d = (mwSize)mxGetScalar(prhs[0]);
    if (d < 3 || nrhs != (int)d+7 || nlhs != (int)d+1) {
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:Order",
            "The input and output counts must agree with d >= 3.");
    }
    require_uint32(prhs[1], "n");
    require_uint32(prhs[2], "r");
    if (mxGetNumberOfElements(prhs[1]) != d ||
            mxGetNumberOfElements(prhs[2]) != d+1) {
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:Dimensions",
            "n and r must contain d and d+1 entries, respectively.");
    }
    n = (const uint32_T *)mxGetData(prhs[1]);
    r = (const uint32_T *)mxGetData(prhs[2]);
    if (r[0] != r[d]) {
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:RingRank",
            "The cyclic ranks must satisfy r(1)=r(d+1).");
    }

    cores = (const double **)mxMalloc(d*sizeof(*cores));
    gradients = (double **)mxMalloc(d*sizeof(*gradients));
    for (k = 0; k < d; ++k) {
        mwSize expected = (mwSize)r[k]*(mwSize)r[k+1]*(mwSize)n[k];
        mwSize dimensions[3];
        if (!mxIsDouble(prhs[3+k]) || mxIsComplex(prhs[3+k]) ||
                mxGetNumberOfElements(prhs[3+k]) != expected) {
            mxFree(cores); mxFree(gradients);
            mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:CoreSize",
                "Every core must be a real double array of the expected size.");
        }
        cores[k] = mxGetPr(prhs[3+k]);
        dimensions[0] = (mwSize)r[k];
        dimensions[1] = (mwSize)r[k+1];
        dimensions[2] = (mwSize)n[k];
        plhs[k+1] = mxCreateNumericArray(3, dimensions, mxDOUBLE_CLASS, mxREAL);
        gradients[k] = mxGetPr(plhs[k+1]);
    }

    require_uint32(prhs[3+d], "SizeOmega");
    require_uint32(prhs[4+d], "Omega");
    if (mxGetNumberOfElements(prhs[3+d]) != 1) {
        mxFree(cores); mxFree(gradients);
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:SampleCount",
            "SizeOmega must be a scalar.");
    }
    sample_count = (mwSize)mxGetScalar(prhs[3+d]);
    if (mxGetNumberOfElements(prhs[4+d]) != d*sample_count) {
        mxFree(cores); mxFree(gradients);
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:Indices",
            "Omega must contain d*SizeOmega entries.");
    }
    Omega = (const uint32_T *)mxGetData(prhs[4+d]);
    p = mxGetScalar(prhs[5+d]);
    PA = mxGetPr(prhs[6+d]);
    if (!mxIsDouble(prhs[5+d]) || mxIsComplex(prhs[5+d]) || p <= 0.0 ||
            !mxIsDouble(prhs[6+d]) || mxIsComplex(prhs[6+d]) ||
            mxGetNumberOfElements(prhs[6+d]) != sample_count) {
        mxFree(cores); mxFree(gradients);
        mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:TrainingData",
            "p must be positive and PA must contain SizeOmega real doubles.");
    }

    plhs[0] = mxCreateDoubleMatrix(sample_count, 1, mxREAL);
    values = mxGetPr(plhs[0]);
    prefix = (double **)mxMalloc((d+1)*sizeof(*prefix));
    suffix = (double **)mxMalloc((d+1)*sizeof(*suffix));
    for (k = 0; k <= d; ++k) {
        prefix[k] = (double *)mxMalloc((mwSize)r[0]*(mwSize)r[k]*sizeof(double));
        suffix[k] = (double *)mxMalloc((mwSize)r[k]*(mwSize)r[0]*sizeof(double));
    }
    identity(prefix[0], (mwSize)r[0]);
    identity(suffix[d], (mwSize)r[0]);

    for (sample = 0; sample < sample_count; ++sample) {
        for (k = 0; k < d; ++k) {
            mwSize index = (mwSize)Omega[sample*d+k];
            mwSize slice_entries = (mwSize)r[k]*(mwSize)r[k+1];
            const double *slice;
            if (index < 1 || index > (mwSize)n[k]) {
                for (i = 0; i <= d; ++i) {
                    mxFree(prefix[i]); mxFree(suffix[i]);
                }
                mxFree(prefix); mxFree(suffix);
                mxFree(cores); mxFree(gradients);
                mexErrMsgIdAndTxt("LRTCTR:ComputeGradsGeneral:Indices",
                    "Omega contains an out-of-range index.");
            }
            slice = cores[k]+(index-1)*slice_entries;
            multiply(prefix[k], slice, prefix[k+1], (mwSize)r[0],
                     (mwSize)r[k], (mwSize)r[k+1]);
        }
        values[sample] = 0.0;
        for (i = 0; i < (mwSize)r[0]; ++i) {
            values[sample] += prefix[d][i+i*(mwSize)r[0]];
        }

        for (k = d; k-- > 0;) {
            mwSize index = (mwSize)Omega[sample*d+k];
            mwSize slice_entries = (mwSize)r[k]*(mwSize)r[k+1];
            const double *slice = cores[k]+(index-1)*slice_entries;
            multiply(slice, suffix[k+1], suffix[k], (mwSize)r[k],
                     (mwSize)r[k+1], (mwSize)r[0]);
        }

        {
            double scale = (values[sample]-PA[sample])/p;
            for (k = 0; k < d; ++k) {
                mwSize index = (mwSize)Omega[sample*d+k];
                mwSize offset = (index-1)*(mwSize)r[k]*(mwSize)r[k+1];
                for (j = 0; j < (mwSize)r[k+1]; ++j) {
                    for (i = 0; i < (mwSize)r[k]; ++i) {
                        double environment = 0.0;
                        for (ell = 0; ell < (mwSize)r[0]; ++ell) {
                            environment += suffix[k+1][j+ell*(mwSize)r[k+1]]*
                                           prefix[k][ell+i*(mwSize)r[0]];
                        }
                        gradients[k][offset+i+j*(mwSize)r[k]] +=
                            scale*environment;
                    }
                }
            }
        }
    }

    for (k = 0; k <= d; ++k) {
        mxFree(prefix[k]);
        mxFree(suffix[k]);
    }
    mxFree(prefix);
    mxFree(suffix);
    mxFree(cores);
    mxFree(gradients);
}
