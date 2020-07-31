#include <bits/stdc++.h>
using namespace std;
using ll = long long;
typedef pair<int, int> pii;
#define rep(i, start, n) for (int i = (int)(start); i < (int)(n); ++i)
#define all(a) a.begin(), a.end()

const int MOD = 1e9+7;
const int INF = 1001001001;
int edge[8][8] {0};

int main(){
    int N, M;
    cin >> N >> M;
    rep(i, 0, M){
        int a, b;
        cin >> a >> b;
        a--;
        b--;
        edge[a][b] = 1;
        edge[b][a] = 1;
    }
    int ans = 0;
    vector<int> v(N-1);
    iota(v.begin(), v.end(), 1);
    do {
        int bef = 0;
        bool ok = true;
        for (auto now : v){
            if (edge[bef][now] != 1){
                ok = false;
            }
            bef = now;
        }
        if (ok){
            ans++;
        }
    } while(next_permutation(v.begin(), v.end()));
    cout << ans << endl;
}
