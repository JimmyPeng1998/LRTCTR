#include "mex.h"

/*
 * ComputePxGeneral_mex(d, n, r, core1, ..., cored, SizeOmega, Omega)
 *
 * Evaluate a general-order tensor ring only at the sampled multi-indices.
 * Core k is stored in MATLAB column-major order with size
 * r(k)-by-r(k+1)-by-n(k). Omega is the vectorized transpose of an
 * SizeOmega-by-d index matrix and therefore stores one sample contiguously.
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
                value += A[i + ell*rows]*B[ell + j*inner];
            }
            C[i + j*rows] = value;
        }
    }
}

static void require_uint32(const mxArray *array, const char *name)
{
    if (!mxIsUint32(array) || mxIsComplex(array)) {
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Type",
            "%s must be a real uint32 array.", name);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize d, sample_count, k, sample, max_rank, workspace_size;
    const uint32_T *n;
    const uint32_T *r;
    const uint32_T *Omega;
    const double **cores;
    double *left;
    double *right;
    double *values;

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Outputs",
            "Exactly one output is required.");
    }
    if (nrhs < 8) {
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Inputs",
            "At least three tensor cores are required.");
    }

    d = (mwSize) mxGetScalar(prhs[0]);
    if (d < 3 || nrhs != (int)d + 5) {
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Order",
            "The number of tensor cores must agree with d >= 3.");
    }
    require_uint32(prhs[1], "n");
    require_uint32(prhs[2], "r");
    if (mxGetNumberOfElements(prhs[1]) != d ||
            mxGetNumberOfElements(prhs[2]) != d+1) {
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Dimensions",
            "n and r must contain d and d+1 entries, respectively.");
    }
    n = (const uint32_T *) mxGetData(prhs[1]);
    r = (const uint32_T *) mxGetData(prhs[2]);
    if (r[0] != r[d]) {
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:RingRank",
            "The cyclic ranks must satisfy r(1)=r(d+1).");
    }

    cores = (const double **) mxMalloc(d*sizeof(*cores));
    max_rank = 0;
    for (k = 0; k < d; ++k) {
        mwSize expected = (mwSize)r[k]*(mwSize)r[k+1]*(mwSize)n[k];
        if (!mxIsDouble(prhs[3+k]) || mxIsComplex(prhs[3+k]) ||
                mxGetNumberOfElements(prhs[3+k]) != expected) {
            mxFree(cores);
            mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:CoreSize",
                "Every core must be a real double array of the expected size.");
        }
        cores[k] = mxGetPr(prhs[3+k]);
        if ((mwSize)r[k] > max_rank) {
            max_rank = (mwSize)r[k];
        }
    }
    if ((mwSize)r[d] > max_rank) {
        max_rank = (mwSize)r[d];
    }

    require_uint32(prhs[3+d], "SizeOmega");
    require_uint32(prhs[4+d], "Omega");
    if (mxGetNumberOfElements(prhs[3+d]) != 1) {
        mxFree(cores);
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:SampleCount",
            "SizeOmega must be a scalar.");
    }
    sample_count = (mwSize) mxGetScalar(prhs[3+d]);
    if (mxGetNumberOfElements(prhs[4+d]) != d*sample_count) {
        mxFree(cores);
        mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Indices",
            "Omega must contain d*SizeOmega entries.");
    }
    Omega = (const uint32_T *) mxGetData(prhs[4+d]);

    plhs[0] = mxCreateDoubleMatrix(sample_count, 1, mxREAL);
    values = mxGetPr(plhs[0]);
    workspace_size = (mwSize)r[0]*max_rank;
    left = (double *) mxMalloc(workspace_size*sizeof(*left));
    right = (double *) mxMalloc(workspace_size*sizeof(*right));

    for (sample = 0; sample < sample_count; ++sample) {
        mwSize first_index = (mwSize)Omega[sample*d];
        mwSize entries = (mwSize)r[0]*(mwSize)r[1];
        const double *first_slice;
        double *current = left;
        double *next = right;

        if (first_index < 1 || first_index > (mwSize)n[0]) {
            mxFree(left); mxFree(right); mxFree(cores);
            mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Indices",
                "Omega contains an out-of-range index.");
        }
        first_slice = cores[0] + (first_index-1)*entries;
        for (k = 0; k < entries; ++k) {
            current[k] = first_slice[k];
        }

        for (k = 1; k < d; ++k) {
            mwSize index = (mwSize)Omega[sample*d+k];
            mwSize slice_entries = (mwSize)r[k]*(mwSize)r[k+1];
            const double *slice;
            double *swap;
            if (index < 1 || index > (mwSize)n[k]) {
                mxFree(left); mxFree(right); mxFree(cores);
                mexErrMsgIdAndTxt("LRTCTR:ComputePxGeneral:Indices",
                    "Omega contains an out-of-range index.");
            }
            slice = cores[k] + (index-1)*slice_entries;
            multiply(current, slice, next, (mwSize)r[0],
                     (mwSize)r[k], (mwSize)r[k+1]);
            swap = current; current = next; next = swap;
        }

        values[sample] = 0.0;
        for (k = 0; k < (mwSize)r[0]; ++k) {
            values[sample] += current[k+k*(mwSize)r[0]];
        }
    }

    mxFree(left);
    mxFree(right);
    mxFree(cores);
}
