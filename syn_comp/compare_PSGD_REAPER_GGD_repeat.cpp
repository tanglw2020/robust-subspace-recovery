#include "../util/GGD.h"
#include "../util/PSGD_PSGM.h"
#include "../util/PSGD_IRLS.h"
#include "../util/REAPER.h"

#include <armadillo>
#include <cstdlib>
#include <iostream>

using namespace arma;
using std::cout;
using std::endl;

void parse_Cmdline(const char **argv, double *input) {
  for (int i = 1; i < 12; i += 2) {
    switch (argv[i][1]) {
    case 'D':
      input[0] = atof(argv[i + 1]);
      break;
    case 'd':
      input[1] = atof(argv[i + 1]);
      break;
    case 'N':
      input[2] = atof(argv[i + 1]);
      break;
    case 'r':
      input[3] = atof(argv[i + 1]);
      break;
    case 'n':
      input[4] = atof(argv[i + 1]);
      break;
    case 's':
      input[5] = atof(argv[i + 1]);
      break;
    }
  }
}

void repeat_run_GGD(const int N, const double r, const int D, const int d,
                    const double sigma, const int repeat, double &variance) {

  input_GGD in_data;
  output_GGD out_data;
  const int M = ceil(r * N / (1 - r));
  mat X, O, noise, Xtilde_noise, Y;
  vec cos_phi = zeros<vec>(repeat);

  for (int i = 0; i < repeat; ++i) {
    X = join_vert(randn<mat>(d, N), zeros<mat>(D - d, N));
    O = randn<mat>(D, M);
    noise = sigma * randn<mat>(D, N);
    Xtilde_noise = join_horiz(X + noise, O);

    in_data.X = &Xtilde_noise;
    in_data.d = d;
    GGD_solver(in_data, out_data);

    Y = null((out_data.V).t());
    mat B = Y.rows(0, d - 1);
    B = sum(B % B, 0);

    cos_phi[i] = B.max();
  }
  // variance = sqrt(var(cos_phi));
  variance = (mean(cos_phi));
}



void repeat_run_PSGD(const int N, const double r, const int D, const int d,
                    const double sigma, const int repeat, double &variance) {

  input_PSGD in_data;
  output_PSGD out_data;
  const int M = ceil(r * N / (1 - r));
  mat X, O, noise, Xtilde_noise, Y;
  vec cos_phi = zeros<vec>(repeat);

  for (int i = 0; i < repeat; ++i) {
    X = join_vert(randn<mat>(d, N), zeros<mat>(D - d, N));
    O = randn<mat>(D, M);
    noise = sigma * randn<mat>(D, N);
    Xtilde_noise = join_horiz(X + noise, O);

    in_data.X = &Xtilde_noise;
    in_data.minimize = true;
    DPCP_PSGM(in_data, out_data);
    cos_phi[i] = norm(out_data.b_star.rows(0, d - 1));
  }
  // variance = sqrt(var(cos_phi));
  variance = (mean(cos_phi));
}



void repeat_run_FMS(const int N, const double r, const int D, const int d,
                    const double sigma, const int repeat, double &variance) {

  input_IRLS in_data;
  output_IRLS out_data;
  const int M = ceil(r * N / (1 - r));
  mat X, O, noise, Xtilde_noise, Y;
  vec cos_phi = zeros<vec>(repeat);

  for (int i = 0; i < repeat; ++i) {
    X = join_vert(randn<mat>(d, N), zeros<mat>(D - d, N));
    O = randn<mat>(D, M);
    noise = sigma * randn<mat>(D, N);
    Xtilde_noise = join_horiz(X + noise, O);

    in_data.X = &Xtilde_noise;
    in_data.d = d;
    IRLS_REAPER_solver(in_data, out_data);
    mat B = out_data.B_star.rows(0, d - 1);
    B = sum(B % B, 0);

    cos_phi[i] = B.max();
  }
  // variance = sqrt(var(cos_phi));
  variance = (mean(cos_phi));
}




void repeat_run_IRLS_PSGD(const int N, const double r, const int D, const int d,
                    const double sigma, const int repeat, double &variance) {

  input_IRLS_PSGD in_data;
  output_IRLS_PSGD out_data;
  const int M = ceil(r * N / (1 - r));
  mat X, O, noise, Xtilde_noise, Y;
  vec cos_phi = zeros<vec>(repeat);

  for (int i = 0; i < repeat; ++i) {
    X = join_vert(randn<mat>(d, N), zeros<mat>(D - d, N));
    O = randn<mat>(D, M);
    noise = sigma * randn<mat>(D, N);
    Xtilde_noise = join_horiz(X + noise, O);

    in_data.X = &Xtilde_noise;
    in_data.d = d;
    IRLS_PSGD_solver(in_data, out_data);
    mat B = out_data.U.cols(d, D-1);
    B = B.rows(0, d - 1);
    B = sum(B % B, 0);
    B = sort(B);

    cos_phi[i] = B.max();
  }
  // variance = sqrt(var(cos_phi));
  variance = (mean(cos_phi));
}

int main(int argc, const char **argv) {
  wall_clock timer;
  timer.tic();

  arma_rng::set_seed_random();
  // arma_rng::set_seed(1234);

  if (argc != 13) {
    cout << "Not enough input!" << endl;
    return 1;
  }

  double cmdInput[6];
  parse_Cmdline(argv, cmdInput);

  const int D = (int)(cmdInput[0]);       // 30;
  const int d = (int)(cmdInput[1]);       // 29;
  const int N = (int)(cmdInput[2]);       // 1500;
  const double r = cmdInput[3];           // 0.7;
  const int num_seg = (int)(cmdInput[4]); // 200;
  const double sigma_limit = cmdInput[5]; // 0.1;

  const int M = ceil(r * N / (1 - r));
  const mat X = join_vert(randn<mat>(d, N), zeros<mat>(D - d, N));
  const mat O = randn<mat>(D, M);

  const vec sigma = linspace<vec>(0, sigma_limit, num_seg);
  const vec ratios = linspace<vec>(0, r, num_seg);
  vec cos_phi_PSGD = zeros<vec>(num_seg);
  vec cos_phi_REAPER = zeros<vec>(num_seg);
  vec cos_phi_GGD = zeros<vec>(num_seg);
  vec cos_phi_PSGD_IRLS = zeros<vec>(num_seg);
  vec cos_phi_PSGD_IRLS_c1 = zeros<vec>(num_seg);
  mat noise, Xtilde_noise, Y;

  input_PSGD in_PSGD;
  output_PSGD out_PSGD;
  input_IRLS in_IRLS;
  output_IRLS out_IRLS;
  input_GGD in_GGD;
  output_GGD out_GGD;
  input_IRLS_PSGD in_IRLS_PSGD;
  output_IRLS_PSGD out_IRLS_PSGD;
  timer.tic();
  for (int i = 0; i < num_seg; ++i) {
      float rt = ratios(i);
      int M_cur = ceil(rt * N / (1 - rt));
      // mat O_cur = randn<mat>(D, M_cur);
      // noise = sigma_limit * randn<mat>(D, N);
      // Xtilde_noise = join_horiz(X+noise, O_cur);

      // in_PSGD.X = &Xtilde_noise;
      // in_PSGD.minimize = true;
      // DPCP_PSGM(in_PSGD, out_PSGD);
      // cos_phi_PSGD[i] = norm(out_PSGD.b_star.rows(0, d - 1));

      repeat_run_PSGD(N, rt, D, d, sigma_limit, 100, cos_phi_PSGD[i]);

    }
  cout << timer.toc() << endl;

  timer.tic();
  for (int i = 0; i < num_seg; ++i) {
      float rt = ratios(i);
      int M_cur = ceil(rt * N / (1 - rt));
      mat O_cur = randn<mat>(D, M_cur);

      // noise = sigma_limit * randn<mat>(D, N);
      // Xtilde_noise = join_horiz(X+noise, O_cur);

      // in_IRLS.X = &Xtilde_noise;
      // in_IRLS.d = d;
      // IRLS_REAPER_solver(in_IRLS, out_IRLS);
      // cos_phi_REAPER[i] = norm(out_IRLS.B_star.rows(0, d - 1))/ out_IRLS.B_star.n_cols;

      // mat B = out_IRLS.B_star.rows(0, d - 1);
      // B = sum(B % B, 0);
      // cos_phi_REAPER[i] = B.max();

      repeat_run_FMS(N, rt, D, d, sigma_limit, 100, cos_phi_REAPER[i]);

      // (out_IRLS.B_star.t()*out_IRLS.B_star).print();
  }
  cout <<"FMS:"<<timer.toc() << endl;

  timer.tic();
  for (int i = 0; i < num_seg; ++i) {
      float rt = ratios(i);
      int M_cur = ceil(rt * N / (1 - rt));
      mat O_cur = randn<mat>(D, M_cur);
      noise = sigma_limit * randn<mat>(D, N);
      Xtilde_noise = join_horiz(X+noise, O_cur);

    // in_GGD.X = &Xtilde_noise;
    // in_GGD.d = d;
    // GGD_solver(in_GGD, out_GGD);
    // Y = null((out_GGD.V).t());

    // mat B = Y.rows(0, d - 1);
    // B = sum(B % B, 0);
    // // cout<< B.n_cols<< " "<< B.n_rows << endl;
    // cos_phi_GGD[i] = B.max();
    repeat_run_GGD(N, rt, D, d, sigma_limit, 20, cos_phi_GGD[i]);


    // cos_phi_GGD[i] = norm(Y.rows(0, d - 1))/ Y.n_cols;
    // cos_phi_GGD[i] = norm(out_GGD.V.rows(0, d - 1))/ out_GGD.V.n_cols;
    // (out_GGD.V.t()*out_GGD.V).print();

  }
  cout << timer.toc() << endl;

  timer.tic();
  for (int i = 0; i < num_seg; ++i) {
      float rt = ratios(i);
      int M_cur = ceil(rt * N / (1 - rt));
      mat O_cur = randn<mat>(D, M_cur);
      // noise = sigma_limit * randn<mat>(D, N);
      // Xtilde_noise = join_horiz(X+noise, O_cur);

      // in_IRLS_PSGD.X = &Xtilde_noise;
      // in_IRLS_PSGD.d = d;
      // IRLS_PSGD_solver(in_IRLS_PSGD, out_IRLS_PSGD);
      // // mat B = out_IRLS_PSGD.B_star.rows(0, d - 1);
      // mat B = out_IRLS_PSGD.U.cols(d, D-1);
      // B = B.rows(0, d - 1);
      // B = sum(B % B, 0);
      // B = sort(B);
      // cos_phi_PSGD_IRLS[i] = B.max();

      repeat_run_IRLS_PSGD(N, rt, D, d, sigma_limit, 100, cos_phi_PSGD_IRLS[i]);


      // in_IRLS_PSGD.X = &Xtilde_noise;
      // in_IRLS_PSGD.d = D-1;
      // IRLS_PSGD_solver(in_IRLS_PSGD, out_IRLS_PSGD);
      // // B = out_IRLS_PSGD.B_star.rows(0, d - 1);
      // B = out_IRLS_PSGD.U.cols(d, D-1);
      // B = B.rows(0, d - 1);
      // B = sum(B % B, 0);
      // B = sort(B);
      // cos_phi_PSGD_IRLS_c1[i] = B.max();

      // cos_phi_PSGD_IRLS[i] = norm(out_IRLS_PSGD.B_star.rows(0, d - 1))/ out_IRLS_PSGD.B_star.n_cols;
      // (out_IRLS_PSGD.B_star.t()*out_IRLS_PSGD.B_star).print();
      // (out_IRLS_PSGD.B_star).print();

  }
  cout << timer.toc() << endl;


  cout << "Data generated... Time elapsed " << timer.toc() << "s." << endl;

  cos_phi_REAPER.save("./files/cos_phi_REAPER.ty", raw_ascii);
  cos_phi_PSGD.save("./files/cos_phi_PSGD.ty", raw_ascii);
  cos_phi_GGD.save("./files/cos_phi_GGD.ty", raw_ascii);
  cos_phi_PSGD_IRLS.save("./files/cos_phi_PSGD_IRLS.ty", raw_ascii);
  cos_phi_PSGD_IRLS_c1.save("./files/cos_phi_PSGD_IRLS_c1.ty", raw_ascii);

  cout << "Done!" << endl;

  return 0;
}
