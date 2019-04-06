#include <bits/stdc++.h>
using namespace std;
template<class T> ostream& operator<<(ostream &os, vector<T> V){os << "[ "; for(auto v : V) os << v << " "; return os << "]";}
template<class T> ostream& operator<<(ostream &os, set<T> V){os << "[ "; for(auto v : V) os << v << " "; return os << "]";}
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

const double EPS = 1e-10;

vector<vector<int>> get_input(char *argv){
	ifstream fin;
    fin.open(argv);

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

class node{
	public:

	int y0,y1;
	int y;
	bool leaf;
	int attr_index;

	// ------ adding: --------
	bool is_continuous_split;
	double meadian_split;
	// -----------------------

	vector<node *> child;

	node(){
		this->y0 = 0;
		this->y1 = 0;
		this->y = 0;
		this->leaf = 0;
		this->attr_index = -1;
		this->is_continuous_split = 0;
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
		this->is_continuous_split = n.is_continuous_split;
		this->meadian_split = n.meadian_split;
	}
};

vector<vi> train, test, val;

set<int> categorical_attr = {2,3,5,6,7,8,9,10};
map<int,int> attr_range = {{2,7},{3,4}};

set<int> cont_attr = {0,4,11,12,13,14,15,16,17,18,19,20,21,22};


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
int max_depth = 0;
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

double find_meadian(vi &rem_data, int attr){
	vi tmp; double ans;
	for(auto &i : rem_data){
		tmp.pb(train[i][attr]);
	}
	sort(tmp.begin(), tmp.end());
	
	if(tmp.size()%2 == 1){
		ans = tmp[tmp.size() / 2];
	}
	else{
		int a = tmp.size() / 2;
		ans = tmp[a] + tmp[a-1];
		ans /= 2.0;
	}
	return ans;
}

inline int get_binary(double median,int a){
	double ad = a;
	if(ad < median || abs(ad-median) < EPS) return 0; 
	return 1;
}

bool isContin_separabel(vi &rem_data_child, int a){
	int y[2] = {0,0};
	double med = find_meadian(rem_data_child, a);
	for(auto &i : rem_data_child){
		y[get_binary(med,train[i][a])]++;
	}
	return (y[0] * y[1] > 0);
}

void growNode(node *n,vi &rem_data,vi rem_attr,int depth);

void produce_children(node *n,vi &rem_data,vi rem_attr,int depth){
	double best_IG = 0.0;
	int best_attr = 30;
	double H = entropy(n->y0, n->y1);
	
	for(auto a : rem_attr){
		double IG = H;
		vi y_baccha[2] = {vi(attr_range[a], 0), vi(attr_range[a], 0)};
		double meadian = -1.0;
		if(cont_attr.count(a)) meadian = find_meadian(rem_data,a);

		for(auto i : rem_data){
			if(cont_attr.count(a)){
				y_baccha[train[i][y_in]][get_binary(meadian,train[i][a])]++;
			}
			else y_baccha[train[i][y_in]][train[i][a]]++;
		}

		for(int i=0;i<attr_range[a];i++){
			int baccha_cnt = y_baccha[0][i] + y_baccha[1][i];
			if(baccha_cnt == 0) continue;
			double pi = ((double)baccha_cnt)/rem_data.size();
			IG -= pi * entropy(y_baccha[0][i], y_baccha[1][i]);
		}
		// trace(IG, attr_range[a], a);
		// assert(IG >= 0);
		
		if(best_IG < IG) best_IG = IG, best_attr = a;
		else if((abs(best_IG-IG) < EPS) && a < best_attr) best_attr = a;
	}

	n->attr_index = best_attr;
	bool is_best_cont = cont_attr.count(best_attr);

	int child_cnt = attr_range[best_attr];
	vi rem_attr_child;
	
	n->child = vector<node *> (child_cnt, NULL);
	vector<vi> rem_data_child(child_cnt);
	
	for(auto a : rem_attr) if(a != best_attr) rem_attr_child.push_back(a);	

	if(is_best_cont){
		n->is_continuous_split = 1;
		n->meadian_split = find_meadian(rem_data, best_attr);

		for(auto i : rem_data){
			rem_data_child[get_binary(n->meadian_split, train[i][best_attr])].push_back(i);
		}
	}
	else{
		for(auto i : rem_data){
			rem_data_child[train[i][best_attr]].push_back(i);
		}
	}

	for(int i=0;i<child_cnt;i++){
		n->child[i] = NULL;
		if(rem_data_child[i].size() == 0) continue;
		n->child[i] = new node();

		if(is_best_cont){
			// bool isSeparable = isContin_separabel(rem_data_child[i], best_attr);
			// if(isSeparable) rem_attr_child.push_back(best_attr);
			growNode(n->child[i], rem_data_child[i], rem_attr_child, depth);
			// if(isSeparable) rem_attr_child.pop_back();
		}
		else growNode(n->child[i], rem_data_child[i], rem_attr_child, depth);
	}
}

int misclassified = 0;

void growNode(node *n,vi &rem_data,vi rem_attr,int depth){
	node_cnt++;
	max_depth = max(max_depth, depth);

	n->y = rem_data.size();
	n->y0 = n->y1 =  0;
	for(auto &i : rem_data){
		if(train[i][y_in] == 0) n->y0++;
		else n->y1++;
	}
	if(n->y0 == 0 || n->y1 == 0 || rem_attr.size() == 0){
		n->leaf = 1;
		// -------------- debug ------------------
		misclassified += min(n->y0, n->y1);
		// if(rem_attr.size() == 0){
		// 	if(n->y0 > n->y1) misclassified += n->y1;
		// 	else misclassified += n->y0;
		// }
		// ---------------------------------------
		return;
	}

	produce_children(n,rem_data,rem_attr,depth+1);
}

void train_it(){
	vi rem_data(train.size());
	vi rem_attr(attr_cnt);
	for(int i=0;i<attr_cnt;i++) rem_attr[i] = i;
	for(int i=0;i<rem_data.size();i++) rem_data[i] = i;
	root = new node();
	growNode(root,rem_data,rem_attr,0);
}

double test_it(vector<vi> &data){
	int correct_pred;
	correct_pred = 0;
	for(auto &e : data){
		node* n = root;
		int pred = 0;
		while(1){
			// trace(dd, n->attr_index, n->leaf);
			node *baccha;
			if(n->leaf == 0){
				if(n->is_continuous_split) baccha = n->child[get_binary(n->meadian_split, e[n->attr_index])];
				else baccha = n->child[e[n->attr_index]];
			}
			if(n->leaf || baccha == NULL){
				if(n->y0 > n->y1) pred = 0;
				else pred = 1;
				break;
			}
			n = baccha;
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

    preprocessing();

    train_it();
    trace("train done", node_cnt, max_depth, misclassified, train.size());

    cout<<"train acc : "<<test_it(train)<<endl;
    cout<<"test acc : "<<test_it(test)<<endl;
    cout<<"valid acc : "<<test_it(val)<<endl;
}