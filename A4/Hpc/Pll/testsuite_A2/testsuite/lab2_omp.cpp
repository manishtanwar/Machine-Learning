////////////////////////////////////////////////////////
// ---------------------- Matrix.h ------------------///
////////////////////////////////////////////////////////

#ifndef MATRIX_H
#define MATRIX_H
#pragma GCC optimize ("O3")

#include<bits/stdc++.h>
using namespace std;
template<class T> ostream& operator<<(ostream &os, vector<T> V){for(auto v : V) os << v << " "; return os << "\n";}

typedef double ld;
typedef pair<ld,ld> pld;

static ld eps = 1e-5;
static ld eps_new = 1e-10;
inline bool eq(ld x,ld y) {return abs(x-y) < eps_new;}
inline bool gt(ld x,ld y) {return x > y + eps_new;}
inline bool lt(ld x,ld y) {return x < y - eps_new;}

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

namespace GivensRotation{
	pair<Matrix, Matrix> get_qr(Matrix A);
	pair< pld, ld > givens(ld a,ld b);
	pld givensInv(ld rho);
	void qr_algo(Matrix D, Matrix &V, vector<ld> &eig_vals);
}

namespace Jacobi{
	int max_index(int k);
	void update(int k, ld t);
	void qr_algo(Matrix D, Matrix &V, vector<ld> &eig_vals);
}
#endif

////////////////////////////////////////////////////////
// ---------------------- Matrix.cpp ----------------///
////////////////////////////////////////////////////////

inline void throw_error(bool cond, string s){
	if(!cond){
		throw domain_error(s);
	}
}

void Matrix::alloc_space(){
	data = new ld*[m];
	for(int i=0;i<m;i++)
		data[i] = new ld[n];
}

void Matrix::delete_space(){
	if(data == NULL) return;

	for(int i=0;i<m;i++)
		if(data[i] != NULL) delete[] data[i];
	delete[] data;
}

Matrix::Matrix(){
	m = n = 0;
	data = NULL;
}

Matrix::Matrix(int row, int col, ld* data){
	m = row;
	n = col;
	alloc_space();

	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			this->data[i][j] = data[i*n + j];
}

Matrix::Matrix(int row, int col, ld** data){
	m = row;
	n = col;
	alloc_space();

	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			this->data[i][j] = data[i][j];
}

Matrix::Matrix(int row, int col){
	m = row;
	n = col;
	alloc_space();
}

Matrix::Matrix(int row, int col, int z_or_i){
	m = row;
	n = col;
	alloc_space();
	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++)
			data[i][j] = z_or_i * ((int)i==j);
}


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

Matrix& Matrix::operator += (const Matrix &rhs){
	// throw_error(rhs.m == m && rhs.n == n, "+=, dimension error!");

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] += rhs.data[i][j];
	}
	return *this;
}

Matrix& Matrix::operator -= (const Matrix &rhs){
	// throw_error(rhs.m == m && rhs.n == n, "-=, dimension error!");

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] -= rhs.data[i][j];
	}
	return *this;
}

Matrix& Matrix::operator *= (ld rhs){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++)
			data[i][j] *= rhs;
	}
	return *this;
}


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

Matrix& Matrix::operator *= (const Matrix &rhs){
	// throw_error(n == rhs.m, "*=, dimension error");
	Matrix ans(m, rhs.n);
	// Matrix rhsT(rhs);
	// rhsT = rhsT.transpose();

	for(int i=0;i<ans.m;i++){
		for(int j=0;j<ans.n;j++){
			ans.data[i][j] = 0;
			for(int k=0;k<n;k++)
				ans.data[i][j] += data[i][k] * rhs.data[k][j];
		}
	}
	return (*this = ans);
}

Matrix operator + (const Matrix &a, const Matrix &b){
	Matrix c(a);
	c += b;
	return c;
}

Matrix operator - (const Matrix &a, const Matrix &b){
	Matrix c(a);
	c -= b;
	return c;
}

Matrix operator * (const Matrix &a, ld b){
	Matrix c(a);
	c *= b;
	return c;
}

Matrix operator * (const Matrix &a, const Matrix &b){
	Matrix c(a);
	c *= b;
	return c;
}

Matrix Matrix::transpose(){
	Matrix a(n,m);
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			a.data[i][j] = data[j][i];
	return a;
}

pair<Matrix, Matrix> GivensRotation::get_qr(Matrix A){
	ld c,s,rho;
	int m = A.m;
	int n = A.n;

	for(int j1 = 1; j1 <= min(n,m-1); j1++){
		int j = j1 - 1;
		for(int i1 = m; i1 >= j1+1; i1--){
			int i = i1 - 1;
			pld pp;
			tie(pp, rho) = givens(A.data[i-1][j], A.data[i][j]);
			c = pp.first;
			s = pp.second;
			// need to change the following for parallization(not efficient):
			// --------------
			#pragma omp parallel for
			for(int k = j; k < n; k++){
				ld ele1,ele2;
				ele1 = c * A.data[i-1][k] - s * A.data[i][k];
				ele2 = s * A.data[i-1][k] + c * A.data[i][k];
				A.data[i-1][k] = ele1;
				A.data[i][k]   = ele2;
			}
			// --------------
			A.data[i][j] = rho;
		}
	}

	// Q is indentity
	Matrix Q(A.m,A.n,1);

	// Finding Q:
	for(int j1=min(n,m-1); j1 >= 1; j1--){
		int j = j1 - 1;
		for(int i1 = j1+1; i1 <= m; i1++){
			int i = i1 - 1;
			tie(c,s) = givensInv(A.data[i][j]);
			// need to change the following for parallization(not efficient):
			// --------------
			for(int k = j; k < n; k++){
				ld ele1,ele2;
				ele1 = c * Q.data[i-1][k] + s * Q.data[i][k];
				ele2 = -s * Q.data[i-1][k] + c * Q.data[i][k];
				Q.data[i-1][k] = ele1;
				Q.data[i][k]   = ele2;
			}
			// --------------
		}
	}
	Matrix R(A);
	for(int i=0;i<m;i++)
		for(int j=0;j<i;j++)
			R.data[i][j] = 0;
	
	return {Q,R};
}
pair< pld, ld > GivensRotation::givens(ld a,ld b){
	ld c,s,rho;

	if(eq(b,0))
		c = 1, s = 0, rho = 0;
	else if(eq(a,0))
		c = 0, s = 1, rho = 1;
	else{
		if(gt(abs(b), abs(a))){
			ld tau = -a/b;
			s = 1.0 / sqrt(1.0 + tau*tau);
			c = s*tau;
			rho = 2.0 / c;
		}
		else{
			ld tau = -b/a;
			c = 1.0 / sqrt(1.0 + tau*tau);
			s = c*tau;
			rho = s / 2.0;
		}
	}
	return {{c,s},rho};
}

pld GivensRotation::givensInv(ld rho){
	ld c,s;
	if(eq(rho,0))
		c = 1, s = 0;
	else if(eq(rho,1))
		c = 0, s = 1;
	else if(gt(abs(rho),2))
		c = 2.0 / rho, s = sqrt(1.0 - c*c);
	else
		s = 2.0 * rho, c = sqrt(1.0 - s*s);
	return {c,s};
}

void checker_qr(Matrix Q, Matrix R, Matrix D){
	Matrix DT = Q * R;
	#pragma omp parallel for
	for(int i=0;i<DT.m;i++)
		for(int j=0;j<DT.n;j++)
			if(abs(DT.data[i][j] - D.data[i][j]) > 1e-3){
				// trace(Q,R,D,DT);
				cout << i << ' ' << j << endl;
				cout << DT.data[i][j] << ' ' << D.data[i][j] << endl;
				throw domain_error("Wrong QR!!!");
			}

}

void GivensRotation::qr_algo(Matrix D, Matrix &V, vector<ld> &eig_vals){
	// Matrix D(DD);
	Matrix E(D.m, D.n, 1); // Identity
	// D.m = D.n
	Matrix Q,R;
	Matrix D1(D);

	int iter_cnt = 0;
	// cout << fixed << setprecision(10);

	while(1){	
		tie(Q,R) = GivensRotation::get_qr(D);
		// checker_qr(Q,R,D);
		D = R*Q;
		E *= Q;
		ld diff = 0.;
		#pragma omp parallel for
		for(int i=0;i<D.m;i++){
			#pragma omp critical
			diff = max(diff, abs(D1.data[i][i] - D.data[i][i]));
		}
		iter_cnt++;
		// trace(diff);
		if(diff < eps) break;
		D1 = D;
	}
	// trace(iter_cnt);
	checker_qr(Q,R,D1);

	vector< pair<ld,int> > eig_vals_p;
	
	// #pragma omp parallel for
	for(int i = 0; i < D.m; i++){
		eig_vals_p.push_back({D.data[i][i], i});
	}

	sort(eig_vals_p.begin(), eig_vals_p.end(), greater< pair<ld,int> >	());
	
	#pragma omp parallel for
	for(int i = 0; i < D.m; i++){
		int j = eig_vals_p[i].second;
		for(int k = 0; k < D.m; k++)
			V.data[k][i] = E.data[k][j];
	}
	for(int i=0;i<D.m;i++)
		eig_vals.push_back(sqrt(abs(eig_vals_p[i].first)));	
}

ld eps_small = 1e-12;
inline bool eq1(ld x,ld y) {return abs(x-y) == 0;}

void Jacobi::qr_algo(Matrix D, Matrix &V, vector<ld> &eig){
	int n = D.m;
	eig = vector<ld>(n);
	vector<int> index(n);
	vector<bool> changed(n);
	ld y, c, s;
	int state;

	auto maxind = [&](int k)
    {
    	int m = k+1;
    	for(int i=k+2;i<n;i++){
    		if(abs(D.data[k][i]) > abs(D.data[k][m]))
    			m = i;
    	}
        return m;
    };

    auto update = [&](int k, ld t){
    	y = eig[k]; eig[k] = y+t;
    	if(changed[k] && eq1(y, eig[k]))
    		changed[k] = false,
    		state = state - 1;
    	else if(!changed[k] && !eq1(y, eig[k]))
    		changed[k] = true,
    		state = state + 1;
    };

    auto rotate = [&](int k, int l, int i, int j){
    	ld ele1,ele2;
		ele1 = c * D.data[k][l] - s * D.data[i][j];
		ele2 = s * D.data[k][l] + c * D.data[i][j];
		D.data[k][l] = ele1;
		D.data[i][j]   = ele2;
    };

	Matrix E(n,n,1); // Identity
	state = n;
	// trace(state,n);

	for(int k=0;k<n;k++){
		index[k] = maxind(k);
		eig[k] = D.data[k][k];
		changed[k] = true;
	}

	while(state){
		// trace(state);
		int m = 0;
		for(int k=1;k<n-1;k++)
			if(abs(D.data[k][index[k]]) > abs(D.data[m][index[m]]))
				m = k;
		int k = m; int l = index[m];
		ld p = D.data[k][l];
		y = (eig[l]-eig[k]) / 2.0;
		ld d = abs(y) + sqrt(p*p + y*y);
		ld r = sqrt(p*p + d*d);
		c = d / r;
		s = p / r;
		ld t = (p * p) / d;
		if(lt(y,0.0))
			s = -s, t = -t;
		D.data[k][l] = 0.0;
		update(k,-t);
		update(l,t);

		for(int i = 0;i < k; i++)
			rotate(i,k,i,l);
		for(int i = k+1;i < l; i++)
			rotate(k,i,i,l);
		for(int i = l+1; i < n; i++)
			rotate(k,i,l,i);

		for(int i=0;i<n;i++){
			ld ele1,ele2;
			ele1 = c * E.data[i][k] - s * E.data[i][l];
			ele2 = s * E.data[i][k] + c * E.data[i][l];
			E.data[i][k] = ele1;
			E.data[i][l] = ele2;
		}
		index[k] = maxind(k);
		index[l] = maxind(l);
	}
	// V = E;
	// for(auto &z : eig)
	// 	z = sqrt(abs(z));
	// sort(eig.begin(), eig.end(), greater<ld>());
	// trace(eig);

	vector< pair<ld,int> > eig_vals_p;
	for(int i = 0; i < D.m; i++){
		eig_vals_p.push_back({eig[i], i});
	}

	sort(eig_vals_p.begin(), eig_vals_p.end(), greater< pair<ld,int> >	());
	for(int i = 0; i < D.m; i++){
		int j = eig_vals_p[i].second;
		eig[i] = (sqrt(abs(eig_vals_p[i].first)));
		for(int k = 0; k < D.m; k++)
			V.data[k][i] = E.data[k][j];
	}
}

////////////////////////////////////////////////////////
// --------------------- given.cpp ------------------///
////////////////////////////////////////////////////////
#include <malloc.h>
#include <omp.h>

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */

void check(Matrix &DT, Matrix &U, Matrix &V, vector<ld> &eig){
	int n,m;
	n = DT.m; m = DT.n;
	Matrix Sig(n,m,0);
	for(int i=0;i<n;i++)
		Sig.data[i][i] = eig[i];
	Matrix VT = V.transpose();
	
	Matrix DT_ob(DT.m,DT.n);
	DT_ob = U * Sig * VT;

	// cout << DT << DT_ob;
	ld EPS = 1e-6;
	// cout << fixed << setprecision(12);

	for(int i=0;i<DT.m;i++)
		for(int j=0;j<DT.n;j++){
			if(abs(DT.data[i][j] - DT_ob.data[i][j]) > EPS){
				// cout << i << ' ' << j << endl;
				// cout << DT.data[i][j] << ' ' << DT_ob.data[i][j] << endl;
				throw domain_error("Wrong D!!!");
			}
			// cout << DT.data[i][j] << ' ' << DT_ob.data[i][j] << ' ' << abs(DT.data[i][j] - DT_ob.data[i][j]) << endl;
		}
	// trace("here");
}

void SVD1(int m_in, int n_in, float* D_in, float** U_out_p, float** SIGMA_out_p, float** V_T_out_p)
{
	// cout << fixed << setprecision(6);
	float* U_out = *U_out_p;
	float* SIGMA_out = *SIGMA_out_p;
	float* V_T_out = *V_T_out_p;

	int n,m;
	m = m_in; n = n_in;
	Matrix D(m_in, n_in);
	for(int i=0;i<D.m;i++)
		for(int j=0;j<D.n;j++)
			D.data[i][j] = (ld)D_in[D.n*i + j];

	Matrix M = D.transpose();
	Matrix M_T = D;
	Matrix MTM = M_T * M;
	Matrix V(m,m), Sig_inv(m,n,0);
	vector<ld> eig_vals;

	// GivensRotation::qr_algo(MTM, V, eig_vals);
	Jacobi::qr_algo(MTM, V, eig_vals);

	for(int i = 0; i < n; i++)
		if(!eq(eig_vals[i],0)){
			Sig_inv.data[i][i] = 1.0 / eig_vals[i];
		}

	Matrix U;
	U = M * V * Sig_inv;
	// cout << fixed << setprecision(6);

	// // --------- debug ---------------
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
}

void SVD(int m_in, int n_in, float* D_in, float** U_out_p, float** SIGMA_out_p, float** V_T_out_p)
{
	// cout << fixed << setprecision(6);
	float* U_out = *U_out_p;
	float* SIGMA_out = *SIGMA_out_p;
	float* V_T_out = *V_T_out_p;

	int n,m;
	m = m_in; n = n_in;
	Matrix D(m_in, n_in);
	for(int i=0;i<D.m;i++)
		for(int j=0;j<D.n;j++)
			D.data[i][j] = (ld)D_in[D.n*i + j];

	Matrix M = D.transpose();
	Matrix M_T(D);
	Matrix MMT = M * M_T;
	Matrix U(n,n), Sig_invT(n,m,0);
	vector<ld> eig_vals;

	// Jacobi::qr_algo(MMT, U, eig_vals);
	GivensRotation::qr_algo(MMT, U, eig_vals);

	for(int i = 0; i < n; i++)
		if(!eq(eig_vals[i],0)){
			Sig_invT.data[i][i] = 1.0 / eig_vals[i];
		}

	Matrix V;
	// U = M * V * Sig_inv;
	V = M_T * U * Sig_invT;

	// cout << fixed << setprecision(6);
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
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D_in, float* U, float* SIGMA, float** D_HAT, int *K)
{
	double ret = retention / 100.0;

	double sum = 0.0;
	double sum1 = 0.0;
	for(int i=0;i<N;i++)
		sum += SIGMA[i] * SIGMA[i];

	for(int i=0;i<N;i++){
		sum1 += SIGMA[i] * SIGMA[i];
		if(sum1 / sum >= ret){
			*K = i+1; break;
		}
	}

	Matrix W(N, *K);
	for(int i=0;i<W.m;i++)
		for(int j=0;j<W.n;j++)
			W.data[i][j] = (ld)U[i*N+j];

	*D_HAT = (float *)malloc(sizeof(float) * M * (*K));
	Matrix D(M, N);
	for(int i=0;i<D.m;i++)
		for(int j=0;j<D.n;j++)
			D.data[i][j] = (ld)D_in[D.n*i + j];

	Matrix D_HM = D * W;

	for(int i=0;i<D_HM.m;i++)
		for(int j=0;j<D_HM.n;j++)
			(*D_HAT)[i*(*K) + j] = D_HM.data[i][j];

	// cout << *K << '\n';
	// cout << D_HM;
}