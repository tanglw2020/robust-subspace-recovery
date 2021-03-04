#ifndef GGD_H
#define GGD_H

#include <armadillo>
using namespace arma;

typedef struct{
    mat* X;
    int d;
} input_GGD;

typedef struct {
    mat V;
} output_GGD;

typedef struct {
    mat* X;
    mat* V;
    int D;
} input_obj_GGD;

double obj(const input_obj_GGD& in) {
    vec tmp = sum(square((eye<mat>(in.D, in.D) - (*in.V) * (*in.V).t()) * (*in.X)), 0).t();
    // vec tmp = sum(square((*in.V) * (*in.V).t() * (*in.X)), 0).t();
    return sum(sqrt(tmp));
}

double calc_sdist(const mat* s1 , const mat* s2){
    mat A = ((*s1).t()*(*s2));
    mat U, W;
    vec s;
    svd_econ(U, s, W, A);
    s = abs(1-square(s));
    // s.t().print();
    double dist=(sum(s));
    // cout<<"dist:"<<dist<<endl;

    return (dist);
}

void GGD_solver(const input_GGD& in, output_GGD& out) {
    wall_clock timer;
    input_obj_GGD in_obj;
    const double mu_min      = 1e-15;
    const double mu_0        = 1e-3;
    const double tol         = 1e-9;
    const int maxiter        = 100;
    const double alpha       = 1e-3;
    const double beta        = 0.5;
    const int D              = (*in.X).n_rows;
    const int d              = in.d;

    cx_vec eigval;
    cx_mat eigvec;
    eig_gen(eigval, eigvec, (*in.X) * (*in.X).t());

    vec lambda = real(eigval);
    mat V = real(eigvec);
    uvec indices = sort_index(lambda, "descend");
    V = V.cols(indices);
    V = V.cols(0, d-1);
    // V = V.cols(d, D-1);

    // V = randn<mat>(D, d);
    // V = normalise(V);

    in_obj.X        = in.X;
    in_obj.V        = &V;
    in_obj.D        = D;
    double obj_old  = obj(in_obj);
    double mu       = mu_0;
    double grad_norm_square;
    mat V_next, QV, partial, grad, Y, T, U, W;
    vec tmp, tmp2, s;
    uvec ind;
    int i = 0;
    double seq_dist = 1;
    // double ttt = 0;
    // while ( i <= maxiter) {
    // while (mu > mu_min && i <= maxiter) {
    while (seq_dist > tol && i <= maxiter) {
        i++;
        QV = eye<mat>(D, D) - V * V.t();
        // QV = V * V.t();
        tmp = sqrt(sum(square(QV * (*in.X)), 0).t());
        ind = find(tmp > 0);
        Y = (*in.X).cols(ind);
        tmp2 = tmp.rows(ind);
    
        // timer.tic();
        // T = sqrt(repmat(tmp2, 1, D));
        T = (repmat(tmp2, 1, D));
        partial = -Y * ((Y.t() / T) * V);
        // partial = -(*in.X) * (Y.t() / T) * V;
        // ttt += timer.toc();

        grad = QV * partial;
        grad_norm_square = pow(norm(grad, "fro"), 2);
        svd_econ(U, s, W, -grad);

        do {
            if (mu <= mu_min)
                break;

            V_next = V*W*diagmat(cos(s*mu))*W.t() + U*diagmat(sin(s*mu))*W.t();

            // V_next = V*W*diagmat(cos(s*mu)) + U*diagmat(sin(s*mu));
            // V_next = normalise(V_next);

            // V_next = V - mu*grad;
            // V_next = normalise(V_next);

            in_obj.V = &V_next;
            // if (obj(in_obj) <= obj_old - alpha * mu * grad_norm_square)
            if (obj(in_obj) < obj_old)
                break;
            mu *= beta;
        } while (true);

        seq_dist = calc_sdist(&V, &V_next);

        V = V_next;
        in_obj.V = &V;
        obj_old = obj(in_obj);
    }

    cout<< "ite:" << i << endl;

    out.V = V;
}


#endif
