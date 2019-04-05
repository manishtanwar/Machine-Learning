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
typedef vector<int> vi;

class node{
	public:

	int y0,y1;
	int y;
	bool leaf;
	int attr_index;

	vector<node *> child;

	node(){
		this->y0 = 0;
		this->y1 = 0;
		this->y = 0;
		this->leaf = 0;
		this->attr_index = -1;
	}

	node(int y0,int y1,int y,bool leaf,int attr_index){
		this->y0 = y0;
		this->y1 = y1;
		this->y = y;
		this->leaf = leaf;
		this->attr_index = attr_index;
	}

	node(const node &n){
		this->y0 = n.y0;
		this->y1 = n.y1;
		this->y = n.y;
		this->leaf = n.leaf;
		this->attr_index = n.attr_index;
		this->child = n.child;
	}
};

vector<vector<int>> get_input(char *argv){
	ifstream fin;
    fin.open(argv);

	string line;
	vector< vector<int> > mat;
    while(getline(fin, line)){
    	vector<int> tokens;
		stringstream ss(line); 
		string ele;
		while(getline(ss, ele, ' '))
			tokens.push_back(stoi(ele));
		mat.push_back(tokens);
    }
    return mat;
}

vector<vi> train, test, val;

set<int> categorical_attr = {2,3,5,6,7,8,9,10};
map<int,int> attr_range = {{2,7},{3,4}};

void modify(vector<vi> &v){
	for(int i=5;i<11;i++)
		for(int j=0;j<v.size();j++)
			v[j][i] += 2;
	for(int j=0;j<v.size();j++)
		v[j][1] -= 1;
}

void preprocessing(){
	for(int i=0;i<23;i++)
		if(!categorical_attr.count(i)) attr_range[i] = 2;
	for(int i=5;i<11;i++)
		attr_range[i] = 12;
	modify(train);
	modify(test);
	modify(val);
}

node *root;
int node_cnt = 0;
int attr_cnt = 23;
int y_in = 23;

inline double entropy(int a, int b){
	double p1,p2;
	p1 = (double)a/(a+b);
	p2 = 1.0 - p1;
	// trace(p1,p2,a,b);
	double ret = 0;
	if(a > 0) ret -= p1*log2(p1);
	if(b > 0) ret -= p2*log2(p2);
	// trace(ret);
	return ret;
}

void growNode(node *n,vi &rem_data,vi &rem_attr);

void produce_children(node *n,vi &rem_data,vi &rem_attr){
	double best_IG = 0.0;
	int best_attr = 30;
	double H = entropy(n->y0, n->y1);
	
	for(auto a : rem_attr){
		double IG = H;
		vi y_baccha[2] = {vi(attr_range[a], 0), vi(attr_range[a], 0)};

		for(auto i : rem_data)
			y_baccha[train[i][y_in]][train[i][a]]++;

		for(int i=0;i<attr_range[a];i++){
			int baccha_cnt = y_baccha[0][i] + y_baccha[1][i];
			// trace(baccha_cnt);
			if(baccha_cnt == 0) continue;
			double pi = ((double)baccha_cnt)/rem_data.size();
			IG -= pi * entropy(y_baccha[0][i], y_baccha[1][i]);
		}
		// trace(IG, attr_range[a], a);
		// assert(IG >= 0);
		
		if(best_IG < IG) best_IG = IG, best_attr = a;
		else if(best_IG == IG && a < best_attr) best_attr = a;
	}
	assert(best_attr >= 0);

	n->attr_index = best_attr;
	int child_cnt = attr_range[best_attr];
	vi rem_attr_child;
	for(auto a : rem_attr) if(a != best_attr) rem_attr_child.push_back(a);
	
	n->child = vector<node *> (child_cnt);
	
	vector<vi> rem_data_child(child_cnt);
	// trace(node_cnt,best_attr,rem_attr.size());
	
	for(auto i : rem_data){
		// trace(child_cnt, train[i][best_attr],best_attr);
		rem_data_child[train[i][best_attr]].push_back(i);
	}

	for(int i=0;i<attr_range[best_attr];i++){
		n->child[i] = NULL;
		if(rem_data_child[i].size() == 0) continue;
		n->child[i] = new node();
		growNode(n->child[i], rem_data_child[i], rem_attr_child);
	}
}

ofstream fout_train, fout_test, fout_valid;

void test_it_acc(vector<vi> &data, ofstream &fout){
	int correct_pred;
	correct_pred = 0;
	
	// trace(node_cnt);

	for(auto &e : data){
		node* n = root;
		int pred = 0;
		while(1){
			// if(n->leaf == 0 && e.size() <= n->attr_index){
			// 	trace(e.size(), n->attr_index, n->leaf, depth);
			// }
			assert(((n->leaf==1) || e.size() > n->attr_index));
			if(n->leaf || n->child[e[n->attr_index]] == NULL){
				if(n->y0 > n->y1) pred = 0;
				else pred = 1;
				break;
			}
			n = n->child[e[n->attr_index]];
		}
		if(pred == e[y_in]) correct_pred++;
	}
	fout << node_cnt << " " << 100.0 * (double)correct_pred/data.size() << '\n';
}

set<int> done;

void growNode(node *n,vi &rem_data,vi &rem_attr){
	node_cnt++;
	// trace(node_cnt);
	n->y = rem_data.size();
	n->y0 = n->y1 =  0;
	for(auto &i : rem_data){
		if(train[i][y_in] == 0) n->y0++;
		else n->y1++;
	}
	if(n->y0 == 0 || n->y1 == 0 || rem_attr.size() == 0){
		n->leaf = 1;
		return;
	}
	produce_children(n,rem_data,rem_attr);

	if(node_cnt%40 == 0 && !done.count(node_cnt)){
		done.insert(node_cnt);
		test_it_acc(train, fout_train);
		test_it_acc(test, fout_test);
		test_it_acc(val, fout_valid);
	}
}

void train_it(){
	vi rem_data(train.size());
	vi rem_attr(attr_cnt);
	for(int i=0;i<attr_cnt;i++) rem_attr[i] = i;
	for(int i=0;i<rem_data.size();i++) rem_data[i] = i;
	root = new node();
	growNode(root,rem_data,rem_attr);
}

double test_it(vector<vi> &data){
	int correct_pred;
	correct_pred = 0;
	for(auto &e : data){
		node* n = root;
		int pred = 0;
		while(1){
			if(n->leaf || n->child[e[n->attr_index]] == NULL){
				if(n->y0 > n->y1) pred = 0;
				else pred = 1;
				break;
			}
			n = n->child[e[n->attr_index]];
		}
		if(pred == e[y_in]) correct_pred++;
	}
	return 100.0 * (double)correct_pred/data.size();
}

int main(int argc, char *argv[]){
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0); cout<<setprecision(5);

    train = get_input(argv[1]);
    test = get_input(argv[2]);
    val = get_input(argv[3]);
    
    fout_valid.open("accuracy_val");
    fout_test.open("accuracy_test");
    fout_train.open("accuracy_train");

    preprocessing();
    train_it();
    // cout<<node_cnt<<endl;

    cout<<"train acc : "<<test_it(train)<<endl;
    cout<<"test acc : "<<test_it(test)<<endl;
    cout<<"valid acc : "<<test_it(val)<<endl;
}