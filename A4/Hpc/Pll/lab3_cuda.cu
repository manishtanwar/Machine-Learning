#include "lab3_cuda.h"

// ******************** matrix.h ***************************

#ifndef MATRIX_H
#define MATRIX_H
#include<bits/stdc++.h>
using namespace std;
typedef double ld;
typedef pair<ld,ld> pld;

void throw_error(bool cond, string s);
class Matrix{
	public:
	int m, n;
	ld **data;

	void alloc_space();
	void delete_space();

	Matrix();
	Matrix(int row, int col, ld* data);
	Matrix(int row, int col, ld** data);
	Matrix(int row, int col);
	Matrix(int row, int col, int z_or_i);
	Matrix(const Matrix& a);
	~Matrix();

	Matrix transpose();

	Matrix& operator += (const Matrix &rhs);
	Matrix& operator -= (const Matrix &rhs);
	Matrix& operator *= (ld rhs);
	Matrix& operator =  (const Matrix &rhs);

	// Matrix multiplication method
	Matrix& operator *= (const Matrix &rhs);
};

Matrix operator + (const Matrix &a, const Matrix &b);
Matrix operator - (const Matrix &a, const Matrix &b);
Matrix operator * (const Matrix &a, ld b);
Matrix operator * (const Matrix &a, const Matrix &b);

std::ostream &operator<<(std::ostream &os, const Matrix &m);

namespace Jacobi{
	void qr_algo(Matrix D, Matrix &V, vector<ld> &eig_vals);
}

#endif

// ******************* matrix.cu **********************
inline void throw_error(bool cond, string s){
	if(!cond){
		throw domain_error(s);
	}
}

__attribute__((optimize("-O3")))
void Matrix::alloc_space(){

	ld* tmp = new ld[m*n];
	data = new ld*[m];
	for(int i=0;i<m;i++)
		data[i] = tmp + i*n;
}
__attribute__((optimize("-O3")))
void Matrix::delete_space(){
	if(data == NULL) return;

    delete[] data[0];
	delete[] data;
}

Matrix::Matrix(){
	m = n = 0;
	data = NULL;
}

__attribute__((optimize("-O3")))
Matrix::Matrix(int row, int col, ld* data){
	m = row;
	n = col;
	alloc_space();

	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			this->data[i][j] = data[i*n + j];
}

__attribute__((optimize("-O3")))
Matrix::Matrix(int row, int col, ld** data){
	m = row;
	n = col;
	alloc_space();

	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			this->data[i][j] = data[i][j];
}

__attribute__((optimize("-O3")))
Matrix::Matrix(int row, int col){
	m = row;
	n = col;
	alloc_space();
}

__attribute__((optimize("-O3")))
Matrix::Matrix(int row, int col, int z_or_i){
	m = row;
	n = col;
	alloc_space();
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			data[i][j] = z_or_i * ((int)i==j);
}


__attribute__((optimize("-O3")))
Matrix::Matrix(const Matrix& a){
	m = a.m;
	n = a.n;
	alloc_space();
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			data[i][j] = a.data[i][j];
}

Matrix::~Matrix(){
	delete_space();
}

std::ostream &operator<<(std::ostream &os, const Matrix &m){
	os<<"\n";
	for(int i = 0; i < m.m; i++){
		for(int j = 0;j < m.n; j++)
			os<<m.data[i][j]<<' ';
		os<<'\n';
	}
	return os<<"\n";
}

__attribute__((optimize("-O3")))
Matrix& Matrix::operator += (const Matrix &rhs){

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] += rhs.data[i][j];
	}
	return *this;
}

__attribute__((optimize("-O3")))
Matrix& Matrix::operator -= (const Matrix &rhs){

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] -= rhs.data[i][j];
	}
	return *this;
}

__attribute__((optimize("-O3")))
Matrix& Matrix::operator *= (ld rhs){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] *= rhs;
	}
	return *this;
}


__attribute__((optimize("-O3")))
Matrix& Matrix::operator = (const Matrix &rhs){
	if(rhs.m != m || rhs.n != n){
		delete_space();
		m = rhs.m;
		n = rhs.n;
		alloc_space();
	}

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] = rhs.data[i][j];
	}
	return *this;
}

#define BLOCK_SIZE 16

__attribute__((optimize("-O3")))
inline ld* one_d_ptr(ld **data,int m,int n){
	// ld* ptr = new ld[m*n];
	ld *ptr;
	cudaMallocHost((void **)&ptr, sizeof(ld) * m * n);
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			ptr[i*n + j] = data[i][j];
	return ptr;
}

__attribute__((optimize("-O3")))
__global__ void cuda_matmul_kernel(ld *a, ld *b, ld *c, int m, int n, int k){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col<k && row<m){
		double ans = 0.;
		for(int i=0;i<n;i++) ans += a[row*n+i] * b[i*k+col];
		c[row*k+col] = ans;
	}
}

__attribute__((optimize("-O3")))
Matrix& Matrix::operator *= (const Matrix &rhs){
	ld *Ma,*Mb,*Mc;
	cudaMalloc((void **)&Ma, sizeof(ld) * m * n);
	cudaMalloc((void **)&Mb, sizeof(ld) * rhs.m * rhs.n);
	cudaMalloc((void **)&Mc, sizeof(ld) * m * rhs.n);

	cudaMemcpy(Ma, data[0], sizeof(ld) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(Mb, rhs.data[0], sizeof(ld) * rhs.m * rhs.n, cudaMemcpyHostToDevice);
	
	dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 Grid_dim((rhs.n+BLOCK_SIZE-1)/BLOCK_SIZE, (m+BLOCK_SIZE-1)/BLOCK_SIZE);

	cuda_matmul_kernel<<< Grid_dim, Block_dim >>> (Ma,Mb,Mc,m,n,rhs.n);
	Matrix ans(m, rhs.n);
	cudaMemcpy(ans.data[0], Mc, sizeof(ld) * m * rhs.n, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return (*this = ans);
}

__attribute__((optimize("-O3")))
Matrix operator + (const Matrix &a, const Matrix &b){
	Matrix c(a);
	c += b;
	return c;
}

__attribute__((optimize("-O3")))
Matrix operator - (const Matrix &a, const Matrix &b){
	Matrix c(a);
	c -= b;
	return c;
}

__attribute__((optimize("-O3")))
Matrix operator * (const Matrix &a, ld b){
	Matrix c(a);
	c *= b;
	return c;
}

__attribute__((optimize("-O3")))
Matrix operator * (const Matrix &a, const Matrix &b){
	Matrix c(a);
	c *= b;
	return c;
}

__attribute__((optimize("-O3")))
Matrix Matrix::transpose(){
	Matrix a(n,m);
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			a.data[i][j] = data[j][i];
	return a;
}

double **S; //Symmetric matrix (input)
double  *e; //eigenvalues
double **E; //eigenvectors
int  *ind;
bool *changed;
int  state;
int  N;
#define JACOBI_UPDATE_TOLERANCE 0.001

__attribute__((optimize("-O3")))
int maxind(int k) {
    int m = k+1;
    for (int i = k+2; i < N; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }
    return m;
}

__attribute__((optimize("-O3")))
inline void update(int k, double t) {
    double ek_prev = e[k];
    e[k] = ek_prev + t;
    if (e[k] < 0) e[k] = 0;
    if (changed[k] && fabs(ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && fabs(ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

__attribute__((optimize("-O3")))
void init_jacobi() {
    E = (double**)malloc(__SIZEOF_POINTER__*N);
    for (int i=0; i<N; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N);
        for (int j=0; j<N; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N;

    e = (double*)malloc(__SIZEOF_DOUBLE__*N);
    ind = (int*)malloc(__SIZEOF_INT__*N);
    changed = (bool*)malloc(sizeof(bool)*N);

    for (int k=0; k<N; k++){
        ind[k]     = maxind(k);
        e[k]       = S[k][k];
        changed[k] = true;
    }
}

__attribute__((optimize("-O3")))
void Jacobi::qr_algo(Matrix D_mat, Matrix &V, vector<ld> &eig){
// void Jacobi(double **input_matrix, int n, 
//             double **eigenvalues, double ***eigenvectors) {
    N = D_mat.m;
    S = D_mat.data;
    init_jacobi();
	eig = vector<ld>(N);

    int iter_cnt = 0;
    while(state != 0){
		iter_cnt++;
        int m = 0;

        for (int k=1; k<N-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k][l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

		ld ele1,ele2;
        for (int i=0; i<k; i++){
			// rotate(i, k, i, l, c, s);
			ele1 = c * S[i][k] - s * S[i][l];
			ele2 = s * S[i][k] + c * S[i][l];
			S[i][k] = ele1;
			S[i][l] = ele2;
		}
        for (int i=k+1; i<l; i++){
			// rotate(k, i, i, l, c, s);
			ele1 = c * S[k][i] - s * S[i][l];
			ele2 = s * S[k][i] + c * S[i][l];
			S[k][i] = ele1;
			S[i][l] = ele2;
		}
        for (int i=l+1; i<N; i++){
			// rotate(k, i, l, i, c, s);
			ele1 = c * S[k][i] - s * S[l][i];
			ele2 = s * S[k][i] + c * S[l][i];
			S[k][i] = ele1;
			S[l][i] = ele2;
		}

        for (int i=0; i<N; i++){
            ele1 = c * E[i][k] - s * E[i][l];
			ele2 = s * E[i][k] + c * E[i][l];
			E[i][k] = ele1;
			E[i][l] = ele2;
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }
    // ---------- debug ----------------
	// cout<<"iter_cnt : "<<iter_cnt<<endl;
	// ---------------------------------

    vector< pair<ld,int> > eig_vals_p;
	for(int i = 0; i < N; i++){
		eig_vals_p.push_back({e[i], i});
	}
    sort(eig_vals_p.begin(), eig_vals_p.end(), greater< pair<ld,int> >());
    
    for(int i = 0; i < N; i++){
		int j = eig_vals_p[i].second;
		eig[i] = (sqrt(fabs(eig_vals_p[i].first)));
		for(int k = 0; k < N; k++)
			V.data[k][i] = E[k][j];
	}
}



// ******************* lab3_cuda *****************************

__attribute__((optimize("-O3")))
void check(Matrix &DT, Matrix &U, Matrix &V, vector<ld> &eig){
	int n,m;
	n = DT.m; m = DT.n;
	Matrix Sig(n,m,0);
	for(int i=0;i<n;i++)
		Sig.data[i][i] = eig[i];
	Matrix VT = V.transpose();
	Matrix DT_ob(DT.m,DT.n);
	DT_ob = U * Sig * VT;
	ld EPS = 1e-6;
	for(int i=0;i<DT.m;i++)
		for(int j=0;j<DT.n;j++){
			if(abs(DT.data[i][j] - DT_ob.data[i][j]) > EPS){
				throw domain_error("Wrong D!!!");
			}
		}
	// cout<<"check done here\n";
}

__attribute__((optimize("-O3")))
void SVD_and_PCA (int M_in, int N_in, double* D_in, double** U_out_p, double** SIGMA_out_p,double** V_T_out_p, int *SIGMAm, 
int *SIGMAn, double** D_HAT, int *K, int retention){
    // cout << fixed << setprecision(6);
	*U_out_p = (double*) malloc(sizeof(double) * N_in*N_in);
	*SIGMA_out_p = (double*) malloc(sizeof(double) * N_in);
	*V_T_out_p = (double*) malloc(sizeof(double) * M_in*M_in);
	
	double* U_out = *U_out_p;
	double* SIGMA_out = *SIGMA_out_p;
	double* V_T_out = *V_T_out_p;

	int n,m;
	m = M_in; n = N_in;
	*SIGMAm = n;
	*SIGMAn = m;
	Matrix D(m, n);
	for(int i=0;i<D.m;i++)
		for(int j=0;j<D.n;j++)
			D.data[i][j] = (ld)D_in[D.n*i + j];

	Matrix M = D.transpose();
	Matrix M_T(D);
	Matrix MMT = M * M_T;
	Matrix U(n,n), Sig_invT(n,m,0);
    vector<ld> eig_vals;


	Jacobi::qr_algo(MMT, U, eig_vals);

	for(int i = 0; i < n; i++)
		if(!(fabs(eig_vals[i]-0) < (1e-9))){
			Sig_invT.data[i][i] = 1.0 / eig_vals[i];
		}

	Matrix V;
	V = M_T * U * Sig_invT;

	// --------- debug ---------------																				
	// Matrix VT = V.transpose();
	// cout << eig_vals;
	// cout << U << V;

	// -------------------------------

	// Filling outputs:
	for(int i=0;i<m;i++)
		for(int j=0;j<m;j++)
			V_T_out[i*m + j] = V.data[j][i];
	
	for(int i=0;i<n;i++){
		SIGMA_out[i] = eig_vals[i];
		for(int j=0;j<n;j++)
			U_out[i*n + j] = U.data[i][j];
	}

	// check(M, U, V, eig_vals);

	double ret = retention / 100.0;

	double sum = 0.0;
	double sum1 = 0.0;
	for(int i=0;i<N_in;i++)
		sum += SIGMA_out[i] * SIGMA_out[i];

	for(int i=0;i<N_in;i++){
		sum1 += SIGMA_out[i] * SIGMA_out[i];
		if(sum1 / sum >= ret){
			*K = i+1; break;
		}
	}

	Matrix W(N_in, *K);
	for(int i=0;i<W.m;i++)
		for(int j=0;j<W.n;j++)
			W.data[i][j] = U.data[i][j];

	*D_HAT = (double *)malloc(sizeof(double) * M_in * (*K));
	Matrix D_HM = D * W;

	for(int i=0;i<D_HM.m;i++)
		for(int j=0;j<D_HM.n;j++)
			(*D_HAT)[i*(*K) + j] = D_HM.data[i][j];

	// cout << *K << '\n';
	// cout << D_HM;
}