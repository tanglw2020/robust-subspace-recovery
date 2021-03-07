#ifndef IRLS_PSGD_H
#define IRLS_PSGD_H

#include <armadillo>

typedef struct{
    mat* X;
    vec* beta;
    int d;
} input_WLS_PSGD;

typedef struct {
    mat P_star;
    mat V;
    mat U;
} output_WLS_PSGD;

typedef struct{
    mat* X;
    int d;
} input_IRLS_PSGD;

typedef struct {
    mat B_star;
    mat U;
} output_IRLS_PSGD;


void WLS_solver(const input_WLS_PSGD& in, output_WLS_PSGD& out) {
    const int D = (*in.X).n_rows;
    const mat C = (*in.X) * diagmat(*in.beta) * (*in.X).t();
    vec v = zeros<vec>(D);

    cx_vec eigval;
    cx_mat eigvec;
    eig_gen(eigval, eigvec, C);

    vec lambda = real(eigval);
    mat U = real(eigvec);
    uvec indices = sort_index(lambda, "descend");
    lambda = sort(lambda, "descend");
    U = U.cols(indices);
    out.U = U;

    U = U.cols(in.d, D-1);
    // out.P_star = U * U.t();
    out.V = U;
}


void IRLS_PSGD_solver(const input_IRLS_PSGD& in, output_IRLS_PSGD& out) {
    const int D = (*in.X).n_rows;
    const int L = (*in.X).n_cols;
    const double delta = 1e-10;
    const double epsilon = 1e-8;
    const int maxiter = 300;
    double alpha_old = datum::inf;
    vec beta = ones<vec>(L);

    input_WLS_PSGD inWLS;
    inWLS.X = in.X;
    inWLS.d = in.d;
    output_WLS_PSGD outWLS;
    mat temp1;
    vec temp2;
    double alpha_new;
    int i = 0;
    while(true) {
        inWLS.beta = &beta;
        WLS_solver(inWLS, outWLS);
        // temp1 = outWLS.P_star * (*in.X);
        temp1 = outWLS.V.t() * (*in.X);
        temp1 = (sqrt(sum(temp1 % temp1, 0)));
        temp2 = temp1.t();
        // beta = 1 / max(delta*ones<vec>(L), (temp2 % sqrt(sqrt(temp2))));
        beta = 1 / max(delta*ones<vec>(L), (temp2 % ((temp2))));
        alpha_new = sum(temp2);
        if (alpha_new >= alpha_old - epsilon || i++ > maxiter) {
            break;
        }
        alpha_old = alpha_new;
    }
    cout<< "ite:" << i << endl;

    // out.B_star = U.cols(0, D - in.d-1);
    out.B_star = outWLS.V;
    out.U = outWLS.U;
    // cout<< out.U.n_cols << "," << out.U.n_rows << endl;
}


#endif
