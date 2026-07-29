#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>
#include <string_view>

namespace quaml {

class DataFrame;

struct Bar {
    int64_t timestamp;
    double open, high, low, close;
    int64_t volume;
};

struct BarView {
    const DataFrame& df;
    size_t idx;

    int64_t timestamp() const;
    double  open()      const;
    double  high()      const;
    double  low()       const;
    double  close()     const;
    int64_t volume()    const;
    void    advance();
    bool    hasNext()   const;
};

enum class NearestBarType {
    Nearest,
    Forward,
    Backward,
};

class DataFrame {
public:
    std::vector<int64_t> timestamps;
    std::vector<double>  open, high, low, close;
    std::vector<int64_t> volume;

    void readCSV(std::string_view csvPath);
    void readQML(std::string_view qmlPath);
    void saveQML(std::string_view outPath);

    BarView getBarExact(int64_t ts);
    BarView getBarInexact(int64_t ts);
    BarView getBarInexact(int64_t ts, NearestBarType type);
    BarView getNextBar(BarView& thisBar);
    bool    hasNextBar(BarView& thisBar) const;

    std::vector<BarView> getBars(int64_t begin, int64_t end);
};

// BarView inline definitions — placed here so DataFrame is fully defined.
inline int64_t BarView::timestamp() const { return df.timestamps[idx]; }
inline double  BarView::open()      const { return df.open[idx]; }
inline double  BarView::high()      const { return df.high[idx]; }
inline double  BarView::low()       const { return df.low[idx]; }
inline double  BarView::close()     const { return df.close[idx]; }
inline int64_t BarView::volume()    const { return df.volume[idx]; }
inline void    BarView::advance()         { idx++; }
inline bool    BarView::hasNext()   const { return idx + 1 < df.timestamps.size(); }

} // namespace quaml
