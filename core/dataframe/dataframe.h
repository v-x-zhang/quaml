#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstddef>
#include <string_view>

class DataFrame;

struct Bar{ //OHLCV
    int64_t timestamp; //unix ns
    double open;
    double high;
    double low;
    double close;
    int64_t volume;
};

struct BarView{
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

enum class NearestBarType{
    Nearest, //get nearest by dist
    Forward, //get closest by the next ts after this
    Backward, //get closest by the last ts before this
};

class DataFrame{ //Pandas Dataframes do things pretty well.
public:
    std::vector<int64_t> timestamps;
    std::vector<double> open, high, low, close;
    std::vector<int64_t> volume;


    void readCSV(std::string_view csvPath); //takes in a .csv file and parses line by line.
    void readQML(std::string_view qmlPath); //takes in a custom .qml file

    void saveQML(std::string_view outPath); //writes to a specific path

    BarView getBarExact(std::int64_t ts); //gets a specific bar by timestamp
    BarView getBarInexact(std::int64_t ts); //gets nearest bar to this time (NEAREST DEF)
    BarView getBarInexact(std::int64_t ts, NearestBarType type); //gets nearest bar to this time

    BarView getNextBar(BarView& thisBar); //returns the bar after this bar.
    bool hasNextBar(BarView& thisBar) const;

    std::vector<BarView> getBars(std::int64_t begin, std::int64_t end); //gets bars between the time stamps

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