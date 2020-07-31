#include <bits/stdc++.h>
using namespace std;
using ll = long long;
typedef pair<int, int> pii;
#define rep(i, start, n) for (int i = (int)(start); i < (int)(n); ++i)
#define all(a) a.begin(), a.end()
#define mp(a, b) make_pair(a, b)

const int MOD = 1e9+7;
const int INF = 1001001001;

template<typename T>
struct ST {
    vector<T> seg;
    int size;
    ST(int n) {
        size = 1;
        while (size < n) size *= 2;
        seg.resize(2*size-1, 0);
    }
    inline T merge(T x, T y) {
        return max(x, y);
    }
    void update(int k, T a) {
        k += size-1;
        seg[k] = a;
        while (k > 0) {
            k = (k-1)/2;
            seg[k] = merge(seg[k*2+1], seg[k*2+2]);
        }
    }
    T query(int a, int b, int k, int l, int r) {
        if (r <= a || b <= l) return 0;
        if (a <= l && r <= b) return seg[k];
        T vl = query(a, b, k*2+1, l, (l+r)/2);
        T vr = query(a, b, k*2+2, (l+r)/2, r);
        return merge(vl, vr);
    }
    T query(int a, int b) {
        return query(a, b, 0, 0, size);
    }
};

const int MAXN = 100100;
pair<int, int> P[MAXN];

int main() {
    int N;
    cin >> N;
    rep(i, 0, N){
        cin >> P[i].first >> P[i].second;
        P[i].second *= -1;
    }
    sort(P, P+N);
    ST<int> seg(MAXN);
    rep(i, 0, N){
        int w = -P[i].second;
        int ret = seg.query(0, w);
        seg.update(w, ret+1);
    }
    cout << seg.query(0, MAXN) << endl;
}