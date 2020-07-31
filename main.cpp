#include <bits/stdc++.h>
using namespace std;
using ll = long long;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
#define rep(i, start, n) for (int i = (int)(start); i < (int)(n); ++i)
#define all(a) a.begin(), a.end()
#define mp(a, b) make_pair(a, b)

int INF = 100100100;
int MOD = 10e7+9;

int main(){
    int N;
    cin >> N;
    vector<int> a(N, 0);
    ll b = 0;
    rep(i, 0, N){
        cin >> a[i];
        b ^= a[i];
    }
    rep(i, 0, N){
        ll ans = b^a[i];
        cout << ans << ' ';
    }
}