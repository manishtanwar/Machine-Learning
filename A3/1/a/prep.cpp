#include <bits/stdc++.h>
using namespace std;
template<class T> ostream& operator<<(ostream &os, vector<T> V){os << "[ "; for(auto v : V) os << v << " "; return os << "]";}
template<class L, class R> ostream& operator<<(ostream &os, pair<L,R> P){return os << "(" << P.first << "," << P.second << ")";}
#ifndef ONLINE_JUDGE
#define TRACE
#endif

#ifdef TRACE
#define trace(...) __f(#__VA_ARGS__, __VA_ARGS__)
	template <typename Arg1>
	void __f(const char* name, Arg1&& arg1){ cout << name << " : " << arg1 << endl; }
	template <typename Arg1, typename... Args>
	void __f(const char* names, Arg1&& arg1, Args&&... args){const char* comma = strchr(names + 1, ',');cout.write(names, comma - names) << " : " << arg1<<" | ";__f(comma+1, args...);}
#else
#define trace(...) 1
#endif
#define pb push_back

vector<vector<int>> get_input(ifstream &fin){
	string line;
    getline(fin, line);
    getline(fin, line);
	vector< vector<int> > mat;
    while(getline(fin, line)){
    	vector<int> tokens;
		stringstream ss(line); 
		string ele;
		int first = 1;
		while(getline(ss, ele, ',')){
			if(first)
				{first = 0; continue;}
			tokens.push_back(stoi(ele));
		}
		mat.push_back(tokens);
    }
    return mat;
}

int main(int argc, char const *argv[]){
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0); cout<<setprecision(25);
    ifstream fin;
    fin.open(argv[1]);

    auto mat = get_input(fin);

    vector<int> cont_attr = {1,5,12,13,14,15,16,17,18,19,20,21,22,23};
    set<int> cont_attr_set = {1,5,12,13,14,15,16,17,18,19,20,21,22,23};
    
    vector<double> median(24,0.0);
    // trace(mat.size(), mat[0].size());

    vector<int> tmp;
    for(auto y : cont_attr){
    	int z = y-1;
    	// trace(z);
    	tmp.clear();
    	for(auto &v : mat){
    		tmp.push_back(v[z]);
    	}
    	sort(tmp.begin(), tmp.end());
    	
    	if(tmp.size() & 1){
    		median[z] = tmp[tmp.size() / 2];
    	}
    	else{
    		int a = tmp.size() / 2;
    		median[z] = tmp[a] + tmp[a-1];
    		median[z] /= 2.0;
    	}
    }

    fin.close();
    fin.open(argv[2]);
    auto mat1 = get_input(fin);

    ofstream fout;
    fout.open(argv[3]);
    for(auto &v : mat1){
    	for(int i=0;i<24;i++){
    		if(cont_attr_set.count(i+1)){
    			int pr = 0;
    			if(v[i] > median[i])
    				pr = 1;
    			fout<<pr<<' ';
    		}
    		else{
    			fout<<v[i]<<' ';
    		}
    	}
    	fout<<'\n';
    }
}