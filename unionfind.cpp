#include <bits/stdc++.h>
using namespace std;
using ll = long long;
typedef pair<int, int> pii;
#define rep(i, start, n) for (int i = (int)(start); i < (int)(n); ++i)
#define all(a) a.begin(), a.end()

const int MOD = 1e9+7;
const int INF = 1001001001;

class UnionFind{
    vector<ll> par, rank;
public:
    void init(ll n){
        par.resize(n);
        rank.resize(n);
        fill(par.begin(), par.end(), 1);
        fill(rank.begin(), rank.end(), 0);
    }
    ll getsize(ll x){
        return par[find(x)];
    }
    ll find(ll x){
        if (par[x] > 0) return x;
        return -(par[x] = -find(-par[x]));
    }
    void merge(ll x, ll y){
        x = find(x);
        y = find(y);
        if (x == y) return;
        if (rank[x] < rank[y]){
            par[y] += par[x];
            par[x] = -y;
        }
        else{
            par[x] += par[y];
            par[y] = -x;
        }
        if (rank[x] == rank[y]) rank[x]++;
    }
    bool isSame(ll x, ll y){
        return find(x) == find(y);
    }
};

int main(){
    int N, M;
    cin >> N >> M;
    UnionFind uf;
    set<ll> s;
    uf.init(N+1);   /*1-indexed*/
    rep(i, 0 ,M){
        int a, b;
        cin >> a >> b;
        if (uf.isSame(a, b)){
            uf.merge(0, a);
        }
        uf.merge(a, b);
    }
    rep(i, 0, N+1){
        s.insert(uf.find(i));
    }
    cout << s.size()-1 << endl;
}