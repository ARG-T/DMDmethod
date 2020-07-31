#include <bits/stdc++.h>
using namespace std;
using ll = long long;
typedef pair<int, int> pii;
#define rep(i, start, n) for (int i = (int)(start); i < (int)(n); ++i)
#define all(a) a.begin(), a.end()

const int MOD = 1e9+7;
const int INF = 1001001001;

int R, C;
char maze[50][50];

int bfs(int sx, int sy, int gx, int gy){
    int dist[50][50];
    fill(dist[0], dist[50], INF);
    dist[sx][sy] = 0;
    vector<int> dx = {1, -1, 0, 0};
    vector<int> dy = {0, 0, 1, -1};
    deque<pii> q;
    q.emplace_back(sx, sy);
    while (!q.empty()){
        pii xy = q.front();
        int x = xy.first;
        int y = xy.second;
        q.pop_front();
        if (x == gx && y == gy){
            return dist[x][y];
        }
        rep(i, 0, 4){
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (0 <= nx && nx < R && 0 <= ny && ny < C && dist[nx][ny] == INF && maze[nx][ny] == '.'){
                dist[nx][ny] = dist[x][y] + 1;
                q.emplace_back(nx, ny);
            }
        }
    }
    return -1;
}

int main(){
    cin >> R >> C;
    int sx, sy;
    int gx, gy;
    cin >> sx >> sy >> gx >> gy;
    sx--; sy--; gx--; gy--;
    rep(x, 0, R){
        rep(y, 0, C){
            cin >> maze[x][y];
        }
    }
    cout << bfs(sx, sy, gx, gy) << endl;
}