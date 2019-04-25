#include <iostream>
#include "lab2_io.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

void read_matrix (const char* input_filename, int* M, int* N, float** D){
	FILE *fin = fopen(input_filename, "r");
  if (!fin) {
    fprintf(stderr, "Panic! file %s does not exist!", input_filename);
    exit(1);
  }
	fscanf(fin, "%d%d", M, N);
	
	int num_elements = (*M) * (*N);
	*D = (float*) malloc(sizeof(float)*(num_elements));
	
	for (int i = 0; i < num_elements; i++){
		fscanf(fin, "%f", (*D + i));
	}
	fclose(fin);
}

MatrixXf ptr_to_mat(int M, int N, float *ptr)
{
  MatrixXf ret(M,N);
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
      ret(i,j) = *(ptr+i*N+j);
  return ret;
}

pair<bool, float> compare_under_tolerance(MatrixXf const& m, MatrixXf const& n, float tolerances[], size_t t_sz)
{
  assert(m.rows() == n.rows());
  assert(m.cols() == n.cols());
  float m_norm = m.norm();
  float n_norm = n.norm();

  float m_norm_abs = fabs(m_norm);
  float n_norm_abs = fabs(n_norm);
  float d = fabs(m_norm_abs-n_norm_abs)*100.0f/n_norm_abs;
  for (size_t i = 0; i < t_sz; ++i) {
    if (d <= tolerances[i]) {
      return make_pair(true, tolerances[i]);
    }
  }
  return make_pair(false, 100.0f);
}

void write_result (int M, 
		int N, 
		float* D, 
		float* U, 
		float* SIGMA, 
		float* V_T,
		int K, 
		float* D_HAT,
		double pca_computation_time,
		double total_computation_time,
		const char* dhat_fname)
{
  float tolerances[] = { 0.001, 0.01, 0.1, 1.0 };

  MatrixXf Dm = ptr_to_mat(M, N, D);
  MatrixXf Um = ptr_to_mat(N, N, U);
  MatrixXf Vm = ptr_to_mat(M, M, V_T);
  MatrixXf sigma = MatrixXf::Zero(N,M);
  for (size_t i = 0; i < N; ++i)
    sigma(i, i) = *(SIGMA+i);

  MatrixXf Dm_n = Um*sigma*Vm;
  Dm_n.transposeInPlace();

  auto p = compare_under_tolerance(Dm, Dm_n, tolerances, sizeof(tolerances)/sizeof(float));
  bool d_equal = p.first;
  float d_tolerance = p.second;

  float* dhat_correct = nullptr;
  int Mh, Nh;
  read_matrix (dhat_fname, &Mh, &Nh, &dhat_correct);
  assert(Mh == M);
  bool K_equal = (Nh == K);
  bool d_h_equal = false;
  float d_h_tolerance = 100.0f;
  if (K_equal) {
    MatrixXf Dhatm = ptr_to_mat(M, K, D_HAT);
    MatrixXf Dhatm_correct = ptr_to_mat(Mh, Nh, dhat_correct);
    auto p = compare_under_tolerance(Dhatm_correct, Dhatm, tolerances, sizeof(tolerances)/sizeof(float));
    d_h_equal = p.first;
    d_h_tolerance = p.second;
  }

#define FLAG(f) ((f) ? 'T' : 'F')
  cout << FLAG(d_equal) << ", " << d_tolerance << ", " << FLAG(K_equal) << ", " << K << ", " << FLAG(d_h_equal) << ", " << d_h_tolerance << ", " << pca_computation_time <<  ", " << total_computation_time << endl;
}


