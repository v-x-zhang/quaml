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

    int64_t timestamp() const { return df.timestamps[idx]; }
    double  open()      const { return df.open[idx]; }
    double  high()      const { return df.high[idx]; }
    double  low()       const { return df.low[idx]; }
    double  close()     const { return df.close[idx]; }
    int64_t volume()    const { return df.volume[idx]; }
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

    std::vector<BarView> getBars(std::int64_t begin, std::int64_t end); //gets bars between the time stamps

};