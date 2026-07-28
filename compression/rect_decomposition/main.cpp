#include <cstdlib>
#include <vector>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <fstream>

static const double PI = std::acos(-1.0);

struct Rect{
    std::pair<double, double> center;
    char angle;   // tilt of vertical axis from true vertical, [-45,+45) deg mapped to [-128,127]
    bool topLong; // true if the top/bottom edges (horizontal-ish) are the longer pair
    double longLength;
    double shortLength;
};

double cart_dist(const std::pair<double, double>& p1, const std::pair<double, double>& p2){
    return std::sqrt((p2.first - p1.first) * (p2.first - p1.first) + (p2.second - p1.second) * (p2.second - p1.second));
}

// Returns angle in degrees of vector (dx,dy) relative to positive Y axis, in [-90, 90)
static double axisAngleDeg(double dx, double dy) {
    // atan2 gives angle from +X; we want angle from +Y
    double deg = std::atan2(dx, dy) * 180.0 / PI; // [-180, 180]
    // Normalize to [-90, 90) by folding: opposite directions are the same axis
    if (deg >= 90.0)  deg -= 180.0;
    if (deg < -90.0)  deg += 180.0;
    return deg;
}

// Encode axis angle [-45,+45) degrees into signed char [-128,127]
static char encodeAngle(double deg) {
    return static_cast<char>(std::round(deg / 45.0 * 127.0));
}

// Grid spatial index: maps cell (gx, gy) -> list of point indices in that cell.
// Cell size = cornerTol, so a lookup checks the 3x3 neighborhood for all candidates.
struct GridIndex {
    double cellSize;
    std::unordered_map<long long, std::vector<int>> cells;

    explicit GridIndex(double tol) : cellSize(tol) {}

    long long key(int gx, int gy) const {
        return (long long)gx * 2000003LL + gy;
    }

    void insert(int idx, double x, double y) {
        int gx = (int)std::floor(x / cellSize);
        int gy = (int)std::floor(y / cellSize);
        cells[key(gx, gy)].push_back(idx);
    }

    // Returns index of point closest to (tx,ty) within cellSize distance, excluding `exclude` set.
    int lookup(const std::vector<std::pair<double,double>>& pts,
               double tx, double ty,
               int ex0, int ex1, int ex2) const {
        int gx = (int)std::floor(tx / cellSize);
        int gy = (int)std::floor(ty / cellSize);
        int best = -1;
        double bestDist = cellSize + 1.0;
        for (int dgx = -1; dgx <= 1; dgx++) {
            for (int dgy = -1; dgy <= 1; dgy++) {
                auto it = cells.find(key(gx+dgx, gy+dgy));
                if (it == cells.end()) continue;
                for (int idx : it->second) {
                    if (idx == ex0 || idx == ex1 || idx == ex2) continue;
                    double d = std::hypot(pts[idx].first - tx, pts[idx].second - ty);
                    if (d < bestDist) { bestDist = d; best = idx; }
                }
            }
        }
        return (bestDist <= cellSize) ? best : -1;
    }
};

struct RectCandidate {
    int idx[4];   // indices into points
    double score; // lower = closer to perfect rectangle
    Rect rect;
};

std::vector<std::pair<Rect, std::array<int,4>>> getRectangles(const std::vector<std::pair<double, double>>& points){
    double maxLengthDiff = 2.0;
    double maxParallelAngDiff = 10.0;
    double maxPerpendicularAngDiff = 10.0;
    double cornerTol = 1.5;
    // Only consider edges shorter than this — rectangles spanning the whole grid are unlikely
    // to have all 4 corners present, and long edges dominate the O(N^2) pair loop.
    double maxEdgeLen = 10.0;

    int N = (int)points.size();

    // Build spatial index and per-point neighbor lists (within maxEdgeLen)
    GridIndex grid(cornerTol);
    for (int i = 0; i < N; i++) grid.insert(i, points[i].first, points[i].second);

    // Precompute neighbors within maxEdgeLen for each point
    std::vector<std::vector<int>> neighbors(N);
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double d = std::hypot(points[j].first - points[i].first,
                                  points[j].second - points[i].second);
            if (d <= maxEdgeLen) {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
            }
        }
    }

    std::vector<RectCandidate> candidates;

    // For each edge (i,j), use neighbors[i] as candidates for k (the perpendicular point).
    // Then look up the 4th corner via the grid index — O(1) instead of O(N).
    for (int i = 0; i < N; i++) {
        for (int j : neighbors[i]) {
            if (j <= i) continue; // each pair once
            double edgeLen = std::hypot(points[j].first - points[i].first,
                                        points[j].second - points[i].second);
            double dx = points[j].first  - points[i].first;
            double dy = points[j].second - points[i].second;
            double px = -dy / edgeLen; // unit perpendicular (CCW)
            double py =  dx / edgeLen;

            for (int k : neighbors[i]) {
                if (k == j) continue;

                double dkx = points[k].first  - points[i].first;
                double dky = points[k].second - points[i].second;
                double width = dkx * px + dky * py;
                if (std::abs(width) < 1e-6) continue;

                double along = dkx * (dx/edgeLen) + dky * (dy/edgeLen);
                double perpAngleErr = std::abs(std::atan2(along, std::abs(width)) * 180.0 / PI);
                if (perpAngleErr > maxPerpendicularAngDiff) continue;

                double ex4 = points[j].first  + width * px;
                double ey4 = points[j].second + width * py;
                int l = grid.lookup(points, ex4, ey4, i, j, k);
                if (l < 0) continue;

                double side1 = edgeLen;
                double side2 = cart_dist(points[i], points[k]);
                double side3 = cart_dist(points[j], points[l]);
                double side4 = cart_dist(points[k], points[l]);

                double oppDiff1 = std::abs(side1 - side4);
                double oppDiff2 = std::abs(side2 - side3);
                if (oppDiff1 > maxLengthDiff || oppDiff2 > maxLengthDiff) continue;

                double dklx = points[l].first  - points[k].first;
                double dkly = points[l].second - points[k].second;
                double lenKL = std::hypot(dklx, dkly);
                if (lenKL < 1e-6) continue;
                double parallelAngleErr = std::abs(axisAngleDeg(dx, dy) - axisAngleDeg(dklx, dkly));
                if (parallelAngleErr > 90.0) parallelAngleErr = 180.0 - parallelAngleErr;
                if (parallelAngleErr > maxParallelAngDiff) continue;

                double cx = (points[i].first  + points[j].first  + points[k].first  + points[l].first)  / 4.0;
                double cy = (points[i].second + points[j].second + points[k].second + points[l].second) / 4.0;

                bool edgeIsLong = (side1 >= side2);
                double longLen  = edgeIsLong ? side1 : side2;
                double shortLen = edgeIsLong ? side2 : side1;

                double vx = edgeIsLong ? px : (dx / edgeLen);
                double vy = edgeIsLong ? py : (dy / edgeLen);
                double axisDeg = axisAngleDeg(vx, vy);

                bool topLong = edgeIsLong ? (std::abs(dy/edgeLen) < std::abs(dx/edgeLen))
                                          : (std::abs(vy) < std::abs(vx));

                if (axisDeg >= 45.0 || axisDeg < -45.0) continue;

                Rect r;
                r.center     = {cx, cy};
                r.angle      = encodeAngle(axisDeg);
                r.topLong    = topLong;
                r.longLength = longLen;
                r.shortLength = shortLen;

                double score = oppDiff1 + oppDiff2 + perpAngleErr / maxPerpendicularAngDiff * maxLengthDiff;
                candidates.push_back({{i, j, k, l}, score, r});
            }
        }
    }

    // Deduplicate: same 4 indices (any order) -> keep lowest score
    for (auto& c : candidates) std::sort(c.idx, c.idx + 4);
    std::sort(candidates.begin(), candidates.end(),
        [](const RectCandidate& a, const RectCandidate& b){ return a.score < b.score; });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
        [](const RectCandidate& a, const RectCandidate& b){
            return std::equal(a.idx, a.idx+4, b.idx);
        }), candidates.end());

    // Greedy selection: pick best-scoring non-overlapping rectangles
    std::set<int> usedPoints;
    std::vector<std::pair<Rect, std::array<int,4>>> result;
    for (const auto& c : candidates) {
        bool conflict = false;
        for (int idx : c.idx) if (usedPoints.count(idx)) { conflict = true; break; }
        if (conflict) continue;
        for (int idx : c.idx) usedPoints.insert(idx);
        result.push_back({c.rect, {c.idx[0], c.idx[1], c.idx[2], c.idx[3]}});
    }

    return result;
}

// Decode a Rect back to its 4 corner points.
// The vertical axis (perpendicular to long sides) is tilted `angle` degrees from +Y.
// longLength = length of the long sides (top/bottom if topLong, else left/right).
// shortLength = length of the short sides.
static std::vector<std::pair<double,double>> rectToPoints(const Rect& r) {
    double deg = static_cast<double>(r.angle) / 127.0 * 45.0;
    double rad = deg * PI / 180.0;

    // vertical axis unit vector (tilted from +Y)
    double vx = std::sin(rad);
    double vy = std::cos(rad);
    // horizontal axis unit vector (perpendicular, 90° CCW)
    double hx = -vy;
    double hy =  vx;

    // half-extents along each axis
    double halfV, halfH;
    if (r.topLong) {
        // top/bottom edges are long → horizontal span is longLength/2, vertical span is shortLength/2
        halfH = r.longLength  / 2.0;
        halfV = r.shortLength / 2.0;
    } else {
        halfH = r.shortLength / 2.0;
        halfV = r.longLength  / 2.0;
    }

    double cx = r.center.first, cy = r.center.second;
    return {
        {cx + halfH*hx + halfV*vx, cy + halfH*hy + halfV*vy},
        {cx - halfH*hx + halfV*vx, cy - halfH*hy + halfV*vy},
        {cx - halfH*hx - halfV*vx, cy - halfH*hy - halfV*vy},
        {cx + halfH*hx - halfV*vx, cy + halfH*hy - halfV*vy},
    };
}

template<typename T>
static void writeRaw(std::ofstream& f, const T& val) {
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T>
static void readRaw(std::ifstream& f, T& val) {
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
}

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());

    int N = 1000;
    double lower_bound = -100.0;
    double upper_bound = 100.0;

    std::vector<std::pair<double, double>> points;
    std::uniform_real_distribution<double> dis(lower_bound, upper_bound);

    for (int i = 0; i < N; i++)
        points.push_back({dis(gen), dis(gen)});

    // --- Find rectangles ---
    auto start = std::chrono::steady_clock::now();
    auto rectResults = getRectangles(points);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start).count();

    // Split into rects and their original point indices
    std::vector<Rect> rectangles;
    std::vector<std::array<int,4>> rectIndices;
    for (const auto& [rect, idxs] : rectResults) {
        rectangles.push_back(rect);
        rectIndices.push_back(idxs);
    }

    // Build used-point set directly from the returned indices
    std::set<int> usedIdx;
    for (const auto& idxs : rectIndices)
        for (int idx : idxs) usedIdx.insert(idx);

    std::vector<std::pair<double,double>> remaining;
    for (int i = 0; i < N; i++)
        if (!usedIdx.count(i)) remaining.push_back(points[i]);

    // --- Write compressed file ---
    // Layout:
    //   uint32_t num_rects
    //   uint32_t num_remaining
    //   per rect: double cx, cy, longLength, shortLength  (4×8=32 bytes)
    //             char angle                              (1 byte)
    //             uint8_t topLong                         (1 byte)
    //             = 34 bytes per rect  vs  64 bytes for 4 raw points
    //   remaining: double x, double y per point (16 bytes each)
    {
        std::ofstream f("comp_data.bin", std::ios::binary);
        if (!f) { std::cerr << "Cannot open comp_data.bin\n"; return 1; }

        uint32_t nRects = (uint32_t)rectangles.size();
        uint32_t nRemain = (uint32_t)remaining.size();
        writeRaw(f, nRects);
        writeRaw(f, nRemain);

        for (const auto& r : rectangles) {
            writeRaw(f, r.center.first);
            writeRaw(f, r.center.second);
            writeRaw(f, r.longLength);
            writeRaw(f, r.shortLength);
            writeRaw(f, r.angle);
            uint8_t tl = r.topLong ? 1 : 0;
            writeRaw(f, tl);
        }
        for (const auto& p : remaining) {
            writeRaw(f, p.first);
            writeRaw(f, p.second);
        }
    }

    // --- Write uncompressed file ---
    {
        std::ofstream f("raw_data.bin", std::ios::binary);
        if (!f) { std::cerr << "Cannot open raw_data.bin\n"; return 1; }
        uint32_t n = (uint32_t)N;
        writeRaw(f, n);
        for (const auto& p : points) { writeRaw(f, p.first); writeRaw(f, p.second); }
    }

    // --- Compression ratio ---
    size_t rawBytes  = sizeof(uint32_t) + N * 2 * sizeof(double);
    size_t compBytes = 2 * sizeof(uint32_t)
                     + rectangles.size() * (4 * sizeof(double) + sizeof(char) + sizeof(uint8_t))
                     + remaining.size()  * 2 * sizeof(double);
    double ratio = static_cast<double>(rawBytes) / static_cast<double>(compBytes);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Points: " << N
              << "  |  Rectangles found: " << rectangles.size()
              << "  |  Remaining points: " << remaining.size()
              << "  |  Rectangle search: " << ms << " ms\n";
    std::cout << "Raw size:        " << rawBytes  << " bytes\n";
    std::cout << "Compressed size: " << compBytes << " bytes\n";
    std::cout << "Compression ratio (raw/comp): " << ratio << "x\n\n";

    // --- Reload compressed file and reconstruct points ---
    std::vector<std::pair<double,double>> reconstructed;
    {
        std::ifstream f("comp_data.bin", std::ios::binary);
        if (!f) { std::cerr << "Cannot open comp_data.bin for reading\n"; return 1; }

        uint32_t nRects, nRemain;
        readRaw(f, nRects);
        readRaw(f, nRemain);

        for (uint32_t i = 0; i < nRects; i++) {
            Rect r;
            readRaw(f, r.center.first);
            readRaw(f, r.center.second);
            readRaw(f, r.longLength);
            readRaw(f, r.shortLength);
            readRaw(f, r.angle);
            uint8_t tl; readRaw(f, tl);
            r.topLong = (tl != 0);
            auto corners = rectToPoints(r);
            for (const auto& c : corners) reconstructed.push_back(c);
        }
        for (uint32_t i = 0; i < nRemain; i++) {
            std::pair<double,double> p;
            readRaw(f, p.first);
            readRaw(f, p.second);
            reconstructed.push_back(p);
        }
    }

    // --- Sum of distances: match each original point to its nearest reconstructed point ---
    std::vector<bool> matched(reconstructed.size(), false);
    double totalDist = 0.0;
    for (const auto& orig : points) {
        double best = 1e18;
        int bestIdx = -1;
        for (int i = 0; i < (int)reconstructed.size(); i++) {
            if (matched[i]) continue;
            double d = cart_dist(orig, reconstructed[i]);
            if (d < best) { best = d; bestIdx = i; }
        }
        if (bestIdx >= 0) { matched[bestIdx] = true; totalDist += best; }
    }

    std::cout << "Reconstructed " << reconstructed.size() << " points from compressed file.\n";
    std::cout << "Sum of nearest-neighbor distances (original vs reconstructed): "
              << totalDist << "\n";

    // --- Write CSVs for plotting ---
    {
        std::ofstream f("original.csv");
        f << "x,y\n";
        for (const auto& p : points)
            f << p.first << "," << p.second << "\n";
    }
    {
        std::ofstream f("reconstructed.csv");
        f << "x,y\n";
        for (const auto& p : reconstructed)
            f << p.first << "," << p.second << "\n";
    }
    // Each rectangle is 4 corners + repeat of first to close the polygon, blank line between rects
    {
        std::ofstream f("rectangles.csv");
        f << "x,y\n";
        for (const auto& r : rectangles) {
            auto corners = rectToPoints(r);
            for (const auto& c : corners)
                f << c.first << "," << c.second << "\n";
            f << corners[0].first << "," << corners[0].second << "\n"; // close
            f << "\n";
        }
    }

    return 0;
}