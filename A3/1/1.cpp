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

class node{
	public:

	int y0,y1;
	int y;
	bool leaf;
	int attr_index;

	vector<node *> child;

	node(){}

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

int main(){
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0); cout<<setprecision(25);
    
}