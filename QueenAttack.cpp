#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
int __builtin_popcount(int bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}
/* enable __uint128_t in MSVC */
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#endif

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '['; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << ']'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << "[" << p.first << ", " << p.second << "]"; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << "["; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << "]"; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << "["; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << "]"; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) { out << t.stringify(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.8e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.8e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};
/** rand **/
struct Xorshift {
    static constexpr uint32_t M = UINT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next_u64(); }
    inline uint64_t next_u64() { x ^= x << 7; return x ^= x >> 9; }
    inline uint32_t next_u32() { return next_u64() >> 32; }
    inline uint32_t next_u32(uint32_t mod) { return ((uint64_t)next_u32() * mod) >> 32; }
    inline uint32_t next_u32(uint32_t l, uint32_t r) { return l + next_u32(r - l + 1); }
    inline double next_double() { return next_u32() * e; }
    inline double next_double(double c) { return next_double() * c; }
    inline double next_double(double l, double r) { return next_double(r - l) + l; }
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { auto r = rnd.next_u32(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(std::string str, const std::string& delim) { for (char& c : str) if (delim.find(c) != std::string::npos) c = ' '; std::istringstream iss(str); std::vector<std::string> parsed; std::string buf; while (iss >> buf) parsed.push_back(buf); return parsed; }
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif

struct LogTable {
    static constexpr int M = 65536;
    static constexpr int mask = M - 1;
    double l[M];
    LogTable() : l() {
        unsigned long long x = 88172645463325252ULL;
        double log_u64max = log(2) * 64;
        for (int i = 0; i < M; i++) {
            x = x ^ (x << 7);
            x = x ^ (x >> 9);
            l[i] = log(double(x)) - log_u64max;
        }
    }
    inline double operator[](int i) const { return l[i & mask]; }
} log_table;



static constexpr int NMAX = 32;
static constexpr int CMAX = 6;
using Placement = std::array<int, NMAX>;
using Placements = std::array<Placement, CMAX>;

struct Input {
    int N;
    int C;
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<std::pair<int, int>>> pos;
    Input(std::istream& in) {
        in >> N >> C;
        grid.resize(N, std::vector<int>(N));
        in >> grid;
        pos.resize(C);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (grid[i][j] <= 0) continue;
                pos[grid[i][j] - 1].emplace_back(i, j);
            }
        }
    }
};

struct NQueen {

    static constexpr int NMAX = 32;
    const int N;

    std::array<uint64_t, NMAX * NMAX> hash_table{};

    // There is a queen in row r column c: r2c[r] = c
    std::array<int, NMAX> r2c{};

    // Number of queens present on the right ascending line
    std::array<int, NMAX * 2 - 1> ru{}; // (r, c) -> r + c

    // Number of queens present on the right descending line
    std::array<int, NMAX * 2 - 1> rd{}; // (r, c) -> r + N - 1 - c

    // Number of constraint violations
    int cost = 0;

    uint64_t hash = 0;

    NQueen(int N_, Xorshift& rnd) : N(N_) {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                hash_table[(r << 5) + c] = rnd.next_u64();
            }
        }
        for (int r = 0; r < N; r++) {
            r2c[r] = r;
            push(r, r);
        }
    }

    void shuffle(Xorshift& rnd) {
        for (int r = 0; r < N; r++) pop(r, r2c[r]);
        for (int i = N - 1; i >= 1; i--) {
            auto k = rnd.next_u32(i);
            std::swap(r2c[i], r2c[k]);
        }
        for (int r = 0; r < N; r++) push(r, r2c[r]);
    }

    inline void pop(int r, int c) {
        int x = r + c, y = r + N - 1 - c;
        cost -= (--ru[x]);
        cost -= (--rd[y]);
        hash ^= hash_table[(r << 5) + c];
    }

    inline void push(int r, int c) {
        int x = r + c, y = r + N - 1 - c;
        cost += (ru[x]++);
        cost += (rd[y]++);
        hash ^= hash_table[(r << 5) + c];
    }

    // Swap rows r1 and r2
    int swap(int r1, int r2) {
        int pcost = cost;
        int c1 = r2c[r1], c2 = r2c[r2];
        pop(r1, c1);
        pop(r2, c2);
        push(r1, c2);
        push(r2, c1);
        std::swap(r2c[r1], r2c[r2]);
        return cost - pcost;
    }

};

// Minimum weight maximum matching for bipartite graph
template<typename T>
std::pair<T, std::vector<int>> hungarian(std::vector<std::vector<T>>& A) {
    const T infty = std::numeric_limits<T>::max();
    const int N = (int)A.size();
    const int M = (int)A.front().size();
    std::vector<int> P(M), way(M);
    std::vector<T> U(N, 0), V(M, 0), minV;
    std::vector<bool> used;

    for (int i = 1; i < N; i++) {
        P[0] = i;
        minV.assign(M, infty);
        used.assign(M, false);
        int j0 = 0;
        while (P[j0] != 0) {
            int i0 = P[j0], j1 = 0;
            used[j0] = true;
            T delta = infty;
            for (int j = 1; j < M; j++) {
                if (used[j]) continue;
                T curr = A[i0][j] - U[i0] - V[j];
                if (curr < minV[j]) minV[j] = curr, way[j] = j0;
                if (minV[j] < delta) delta = minV[j], j1 = j;
            }
            for (int j = 0; j < M; j++) {
                if (used[j]) U[P[j]] += delta, V[j] -= delta;
                else minV[j] -= delta;
            }
            j0 = j1;
        }
        do {
            P[j0] = P[way[j0]];
            j0 = way[j0];
        } while (j0 != 0);
    }
    return { -V[0], P };
}

struct DistanceMatrix {

    static constexpr int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    static constexpr int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    const Input& input;

    const int N;
    const int C;
    const int V;
    std::vector<std::vector<double>> dmat;  // distance matrix
    std::vector<std::vector<int>> pmat;     // previous node matrix

    DistanceMatrix(const Input& input_) : input(input_), N(input.N), C(input.C), V(N * N) {
        auto wall = make_vector(false, N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (input.grid[i][j] == -1) wall[i][j] = true;
            }
        }
        auto is_inside = [&](int i, int j) {
            return 0 <= i && i < N && 0 <= j && j < N;
        };
        using E = std::pair<double, int>;
        using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;
        dmat = make_vector(1e9, V, V);
        pmat = make_vector(-1, V, V);
        for (int u = 0; u < V; u++) dmat[u][u] = 0;
        for (int si = 0; si < N; si++) {
            for (int sj = 0; sj < N; sj++) {
                if (wall[si][sj]) continue;
                int s = si * N + sj;
                auto& dist = dmat[s];
                auto& prev = pmat[s];
                PQ pq;
                pq.emplace(0, s);
                while (!pq.empty()) {
                    auto [cost, u] = pq.top(); pq.pop();
                    if (dist[u] < cost) continue;
                    int ui = u / N, uj = u % N;
                    for (int d = 0; d < 8; d++) {
                        int vi = ui + di[d], vj = uj + dj[d];
                        int len = 1;
                        while (is_inside(vi, vj) && !wall[vi][vj]) {
                            int v = vi * N + vj;
                            double ncost = cost + sqrt(len);
                            if (chmin(dist[v], ncost)) {
                                pq.emplace(ncost, v);
                                prev[v] = u;
                            }
                            vi += di[d]; vj += dj[d]; len++;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::pair<int, int>> get_path(int si, int sj, int ti, int tj) const {
        int s = si * N + sj, t = ti * N + tj;
        std::vector<std::pair<int, int>> res;
        const auto& prev = pmat[s];
        res.emplace_back(ti, tj);
        while (prev[t] != -1) {
            t = prev[t];
            res.emplace_back(t / N, t % N);
        }
        if (t != s) return {};
        std::reverse(res.begin(), res.end());
        return res;
    }

    std::vector<std::pair<int, int>> get_step_path(int si, int sj, int ti, int tj) const {
        auto path = get_path(si, sj, ti, tj);
        std::vector<std::pair<int, int>> spath;
        spath.push_back(path[0]);
        for (int i = 1; i < (int)path.size(); i++) {
            auto [si, sj] = spath.back();
            auto [ti, tj] = path[i];
            while (si != ti || sj != tj) {
                if (si < ti) si++;
                else if (si > ti) si--;
                if (sj < tj) sj++;
                else if (sj > tj) sj--;
                spath.emplace_back(si, sj);
            }
        }
        return spath;
    }
    
    double get_dist(int si, int sj, int ti, int tj) const {
        return dmat[si * N + sj][ti * N + tj];
    }

    double compute_matching_cost(const Placements& r2c) const {
        auto A = make_vector(0.0, N + 1, N + 1); // 1-indexed
        double total_matching_cost = 0.0;
        for (int k = 0; k < C; k++) {
            const auto& src = input.pos[k];
            for (int i = 0; i < N; i++) {
                auto [si, sj] = src[i];
                for (int j = 0; j < N; j++) {
                    int ti = j, tj = r2c[k][j];
                    A[i + 1][j + 1] = get_dist(si, sj, ti, tj);
                }
            }
            total_matching_cost += hungarian(A).first;
        }
        return total_matching_cost;
    }

    std::vector<std::tuple<int, int, int, int, int>> compute_matching(const Placements& best_r2c) const {
        auto A = make_vector(0.0, N + 1, N + 1);
        std::vector<std::tuple<int, int, int, int, int>> matching; // (color, r1, c1, r2, c2)
        for (int k = 0; k < C; k++) {
            const auto& src = input.pos[k];
            std::vector<std::pair<int, int>> dst;
            for (int j = 0; j < N; j++) {
                dst.emplace_back(j, best_r2c[k][j]);
            }
            for (int i = 0; i < N; i++) {
                auto [si, sj] = src[i];
                for (int j = 0; j < N; j++) {
                    auto [ti, tj] = dst[j];
                    A[i + 1][j + 1] = get_dist(si, sj, ti, tj);
                }
            }
            auto [dcost, a] = hungarian(A);
            for (int i = 0; i < N; i++) {
                auto [si, sj] = src[a[i + 1] - 1];
                auto [ti, tj] = dst[i];
                matching.emplace_back(k, si, sj, ti, tj);
            }
        }
        return matching;
    }

};

struct MultipleNQueen {

    const Input& input;
    const int N;
    const int C;

    Xorshift rnd;

    std::array<uint64_t, CMAX* NMAX* NMAX> hash_table{};
    Placements r2c{};
    std::array<std::array<int, NMAX * 2 - 1>, CMAX> ru{}; // (r, c) -> r + c
    std::array<std::array<int, NMAX * 2 - 1>, CMAX> rd{}; // (r, c) -> r + N - 1 - c
    std::array<std::array<int, NMAX>, NMAX> overlap{};

    int cost = 0;
    uint64_t hash = 0;

    inline void push(int k, int r, int c) {
        const int x = r + c, y = r + N - 1 - c;
        r2c[k][r] = c;
        cost += (ru[k][x]++);
        cost += (rd[k][y]++);
        cost += (overlap[r][c]++);
        hash ^= hash_table[(k << 10) | (r << 5) | c];
    }

    inline int pop(int k, int r) {
        const int c = r2c[k][r], x = r + c, y = r + N - 1 - c;
        cost -= (--ru[k][x]);
        cost -= (--rd[k][y]);
        cost -= (--overlap[r][c]);
        hash ^= hash_table[(k << 10) | (r << 5) | c];
        r2c[k][r] = -1;
        return c;
    }

    int swap(int k, int r1, int r2) {
        int pcost = cost;
        int c1 = pop(k, r1);
        int c2 = pop(k, r2);
        push(k, r1, c2);
        push(k, r2, c1);
        return cost - pcost;
    }

    MultipleNQueen(const Input& input_) : input(input_), N(input.N), C(input.C) {
        for (auto& x : hash_table) {
            x = rnd.next_u64();
        }
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (input.grid[r][c] == -1) {
                    overlap[r][c]++;
                }
            }
        }
        for (int k = 0; k < C; k++) {
            for (int r = 0; r < N; r++) {
                push(k, r, r);
            }
        }
    }

    void kick(int k) {
        for (int r = 0; r < N; r++) r2c[k][r] = pop(k, r);
        for (int i = N - 1; i >= 1; i--) {
            auto j = rnd.next_u32(i);
            std::swap(r2c[k][i], r2c[k][j]);
        }
        for (int r = 0; r < N; r++) {
            push(k, r, r2c[k][r]);
        }
    }

    void kick() {
        for (int k = 0; k < C; k++) kick(k);
    }

    auto run(const DistanceMatrix& dmat, double duration) {
        Timer timer;
        double temp = 0.1;
        int loop = 0;
        double min_matching_cost = 1e9;
        std::unordered_set<uint64_t> seen;
        std::vector<std::pair<double, Placements>> matching_cost_to_placements;
        for (loop = 0;; loop++) {
            int k = rnd.next_u32(C), r1 = rnd.next_u32(N - 1), r2 = rnd.next_u32(N);
            r2 += (r1 == r2);
            int diff = swap(k, r1, r2);
            if (temp * log_table[loop] > -diff) {
                swap(k, r1, r2);
            }
            else {
                if (!cost) {
                    // valid placements
                    if (!seen.count(hash)) {
                        seen.insert(hash);
                        double matching_cost = dmat.compute_matching_cost(r2c);
                        matching_cost_to_placements.emplace_back(matching_cost, r2c);
                        if (chmin(min_matching_cost, matching_cost)) {
                            dump(loop, min_matching_cost);
                        }
                    }
                }
            }
            if (!(loop & 0xFF)) {
                if (timer.elapsed_ms() > duration) break;
            }
#if 1
            if (!(loop & 0xFFFFF)) {
                kick();
                temp = 0.3;
                dump(loop, seen.size());
            }
            else {
                temp -= 0.2 / 0xFFFFF;
            }
#else
            if (!(loop & 0xFFFFF)) {
                temp = temp < 0.2 ? 0.3 : 0.1;
                dump(loop, seen.size());
            }
#endif
        }
        dump(loop);
        return matching_cost_to_placements;
    }

    void print(const std::vector<std::vector<int>>& a) {
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < a[i].size(); j++) {
                std::cerr << format("%2d ", a[i][j]);
            }
            std::cerr << '\n';
        }
        std::cerr << '\n';
    }

};

struct MoveGenerator {

    static constexpr int EMPTY = -1;
    static constexpr int WALL = -2;

    const Input& input;
    const DistanceMatrix& dmat;

    const int N;
    const int C;

    std::vector<int> colors;
    std::vector<std::tuple<int, int, int, int>> matching;
    std::vector<std::vector<int>> grid;
    std::vector<std::tuple<int, int, int, int>> moves;

    mutable std::vector<double> dist;
    mutable std::vector<int> prev;

    mutable std::vector<std::vector<uint8_t>> buf;

    MoveGenerator(const Input& input_, const DistanceMatrix& dmat_)
        : input(input_), dmat(dmat_), N(input.N), C(input.C)
    {
        dist.resize(N * N);
        prev.resize(N * N);
        buf.resize(N, std::vector<uint8_t>(N));
    }

    void initialize(const std::vector<std::tuple<int, int, int, int, int>>& matching_with_color) {
        for (const auto& [k, si, sj, ti, tj] : matching_with_color) {
            matching.emplace_back(si, sj, ti, tj);
            colors.push_back(k);
        }
        grid = input.grid;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (grid[i][j] == 0) {
                    grid[i][j] = EMPTY;
                }
                else if (grid[i][j] == -1) {
                    grid[i][j] = WALL;
                }
            }
        }
        for (int id = 0; id < (int)matching.size(); id++) {
            const auto& [si, sj, ti, tj] = matching[id];
            grid[si][sj] = id;
        }
    }

    bool is_valid_move(const std::vector<std::pair<int, int>>& path) const {
        int si = path[0].first, sj = path[0].second;
        for (int i = 1; i < (int)path.size(); i++) {
            auto [ti, tj] = path[i];
            while (si != ti || sj != tj) {
                if (si < ti) si++;
                else if (si > ti) si--;
                if (sj < tj) sj++;
                else if (sj > tj) sj--;
                if (grid[si][sj] != EMPTY) return false;
            }
        }
        return true;
    }

    bool is_free_path(int id) const {
        const auto& [si, sj, ti, tj] = matching[id];
        auto spath = dmat.get_step_path(si, sj, ti, tj);
        for (int k = 1; k < (int)spath.size(); k++) {
            auto [i, j] = spath[k];
            if (grid[i][j] != EMPTY) return false;
        }
        return true;
    }

    std::vector<int> calc_obstacle_ids(int id) const {
        for (auto& v : buf) std::fill(v.begin(), v.end(), 0);
        std::vector<int> obstacle_ids;
        const auto& [si, sj, ti, tj] = matching[id];
        auto spath = dmat.get_step_path(si, sj, ti, tj);
        buf[si][sj] = 1;
        for (int k = 1; k < (int)spath.size(); k++) {
            auto [i, j] = spath[k];
            buf[i][j] = 1;
            if (grid[i][j] >= 0) obstacle_ids.push_back(grid[i][j]);
        }
        return obstacle_ids;
    }

    void move_free_path(int id) {
        auto& [si, sj, ti, tj] = matching[id];
        auto path = dmat.get_path(si, sj, ti, tj);
        for (int k = 1; k < (int)path.size(); k++) {
            auto [i1, j1] = path[k - 1];
            auto [i2, j2] = path[k];
            moves.emplace_back(i1, j1, i2, j2);
        }
        std::swap(grid[si][sj], grid[ti][tj]);
        si = ti; sj = tj;
    }

    bool dijkstra(int si, int sj, int ti, int tj) const {
        static constexpr int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
        static constexpr int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
        if (grid[ti][tj] != EMPTY) return false;
        auto is_inside = [&](int i, int j) {
            return 0 <= i && i < N && 0 <= j && j < N;
        };
        using E = std::pair<double, int>;
        using PQ = std::priority_queue<E, std::vector<E>, std::greater<E>>;
        std::fill(dist.begin(), dist.end(), 1e9);
        std::fill(prev.begin(), prev.end(), -1);
        int s = si * N + sj;
        PQ pq;
        pq.emplace(0, s);
        dist[s] = 0;
        while (!pq.empty()) {
            auto [cost, u] = pq.top(); pq.pop();
            if (dist[u] < cost) continue;
            int ui = u / N, uj = u % N;
            if (ui == ti && uj == tj) break;
            for (int d = 0; d < 8; d++) {
                int vi = ui + di[d], vj = uj + dj[d];
                int len = 1;
                while (is_inside(vi, vj) && grid[vi][vj] == EMPTY) {
                    int v = vi * N + vj;
                    double ncost = cost + sqrt(len);
                    if (chmin(dist[v], ncost)) {
                        pq.emplace(ncost, v);
                        prev[v] = u;
                    }
                    vi += di[d]; vj += dj[d]; len++;
                }
            }
        }
        int t = ti * N + tj;
        return dist[t] != 1e9;
    }

    bool dijkstra(int id) const {
        const auto& [si, sj, ti, tj] = matching[id];
        return dijkstra(si, sj, ti, tj);
    }

    void dijkstra_move(int id) {
        auto& [si, sj, ti, tj] = matching[id];
        int s = si * N + sj, t = ti * N + tj;
        assert(dist[t] != 1e9);
        std::vector<std::pair<int, int>> path;
        path.emplace_back(ti, tj);
        while (prev[t] != -1) {
            t = prev[t];
            path.emplace_back(t / N, t % N);
        }
        assert(t == s);
        std::reverse(path.begin(), path.end());
        for (int k = 1; k < (int)path.size(); k++) {
            auto [i1, j1] = path[k - 1];
            auto [i2, j2] = path[k];
            moves.emplace_back(i1, j1, i2, j2);
        }
        std::swap(grid[si][sj], grid[ti][tj]);
        si = ti; sj = tj;
    }

    double calc_force_move_cost(int id) {
        // 進路上の障害物をどかしながら移動した時の、余計に掛かるコストを計算する
        // 玉突きは一旦考慮しない
        // 移動順序は考慮しない
        auto obstacle_ids = calc_obstacle_ids(id);
        std::vector<std::tuple<int, int, int, int, int>> backup; // (id, si, sj, ti, tj)
        double cost = 0;
        bool valid = true;
        for (int oid : obstacle_ids) {
            auto& [si, sj, ti, tj] = matching[oid];
            if (buf[ti][tj]) {
                // 障害物の行き先がパスに含まれる
                valid = false;
                break;
            }
            // dijkstra での移動距離と、障害物を無視して最短で移動した場合の距離の差
            if (dijkstra(si, sj, ti, tj)) {
                cost += dist[ti * N + tj] - dmat.get_dist(si, sj, ti, tj);
                backup.emplace_back(oid, si, sj, ti, tj);
                std::swap(grid[si][sj], grid[ti][tj]);
                si = ti; sj = tj;
            }
            else {
                // infinity cost
                valid = false;
                break;
            }
        }
        std::reverse(backup.begin(), backup.end());
        for (const auto& [oid, psi, psj, pti, ptj] : backup) {
            auto& [si, sj, ti, tj] = matching[oid];
            std::swap(grid[psi][psj], grid[pti][ptj]);
            si = psi; sj = psj;
        }
        return valid ? cost : 1e9;
    }

    void force_move(int id) {
        auto obstacle_ids = calc_obstacle_ids(id);
        for (int oid : obstacle_ids) {
            dijkstra(oid);
            dijkstra_move(oid);
        }
        move_free_path(id);
    }

    void run() {
        auto completed = make_vector(false, matching.size());
        while (true) {
            bool update = false;
            for (int id = 0; id < (int)matching.size(); id++) {
                auto [si, sj, ti, tj] = matching[id];
                if (si == ti && sj == tj) completed[id] = true;
                if (completed[id]) continue;
                if (is_free_path(id)) {
                    move_free_path(id);
                    completed[id] = true;
                    update = true;
                }
                else {
                    auto [si, sj, ti, tj] = matching[id];
                    double force_cost = calc_force_move_cost(id);
                    double dijkstra_cost = dijkstra(id) ? dist[ti * N + tj] : 1e9;
                    if (std::min(dijkstra_cost, force_cost) != 1e9) {
                        if (dijkstra_cost < force_cost) {
                            dijkstra_move(id);
                            completed[id] = true;
                            update = true;
                        }
                        else {
                            force_move(id);
                            completed[id] = true;
                            update = true;
                        }
                    }
                }
            }
            if (!update) break;
        }
    }

    bool attack(int si, int sj, int ti, int tj) const {
        if (si == ti || sj == tj) return true;
        int di = abs(si - ti), dj = abs(sj - tj);
        return di == dj;
    }

    double compute_cost() const {
        double cost = 0.0;
        for (const auto& [si, sj, ti, tj] : moves) {
            cost += sqrt(std::max(abs(si - ti), abs(sj - tj)));
        }
        std::vector<std::vector<std::pair<int, int>>> pos(C);
        for (int i = 0; i < (int)matching.size(); i++) {
            int k = colors[i];
            const auto& [si, sj, ti, tj] = matching[i];
            pos[k].emplace_back(si, sj);
        }
        for (int k = 0; k < C; k++) {
            for (int i = 0; i + 1 < N; i++) {
                auto [si, sj] = pos[k][i];
                for (int j = i + 1; j < N; j++) {
                    auto [ti, tj] = pos[k][j];
                    cost += attack(si, sj, ti, tj) * N;
                }
            }
        }
        return cost;
    }

};



int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if 0
    std::ifstream ifs("../../tester/in/2.in");
    std::istream& in = ifs;
    std::ofstream ofs("../../tester/out/2.out");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    const Input input(in);
    const DistanceMatrix dmat(input);
    dump(input.N, input.C);

    MultipleNQueen mnq(input);
    auto matching_cost_to_placements = mnq.run(dmat, 5000);
    dump(matching_cost_to_placements.size());
    std::sort(matching_cost_to_placements.begin(), matching_cost_to_placements.end());

    double mincost = 1e9;
    std::vector<std::tuple<int, int, int, int, int>> best_matching;
    std::vector<std::tuple<int, int, int, int>> best_moves;

    {
        int loop = 0;
        for (const auto& [matching_cost, placements] : matching_cost_to_placements) {
            loop++;
            if (timer.elapsed_ms() > 6000) break;
            const auto& matching = dmat.compute_matching(placements);
            MoveGenerator mgen(input, dmat);
            mgen.initialize(matching);
            mgen.run();
            const double cost = mgen.compute_cost();
            if (chmin(mincost, cost)) {
                best_moves = mgen.moves;
                best_matching = matching;
                dump(timer.elapsed_ms(), loop, matching_cost, cost, mincost);
            }
        }
    }

    Xorshift rnd;
    int loop = 0;
    auto assign(best_matching);
    while (timer.elapsed_ms() < 9000) {
        loop++;
        int i = rnd.next_u32(assign.size()), j;
        do {
            j = rnd.next_u32(assign.size());
        } while (i == j);
        std::swap(assign[i], assign[j]);
        MoveGenerator mgen(input, dmat);
        mgen.initialize(assign);
        mgen.run();
        if (chmin(mincost, mgen.compute_cost())) {
            best_moves = mgen.moves;
            best_matching = assign;
            dump(timer.elapsed_ms(), mincost);
        }
        else {
            std::swap(assign[i], assign[j]);
        }
    }
    dump(loop);

    out << best_moves.size() << '\n';
    for (const auto& [si, sj, ti, tj] : best_moves) {
        out << si << ' ' << sj << ' ' << ti << ' ' << tj << '\n';
    }

    return 0;
}