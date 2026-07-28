#include "dataframe.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

void DataFrame::readCSV(std::string_view csvPath) {
    std::ifstream file;
    file.open(std::string(csvPath));
    if (!file.is_open())
        throw std::runtime_error("readCSV: cannot open file");

    std::string line;
    file >> std::ws;
    std::getline(file, line); // skip header row

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;

        auto next = [&]() -> std::string {
            std::getline(ss, token, ',');
            return token;
        };

        timestamps.push_back(std::stoll(next()));
        open.push_back(std::stod(next()));
        high.push_back(std::stod(next()));
        low.push_back(std::stod(next()));
        close.push_back(std::stod(next()));
        volume.push_back(std::stoll(next()));
    }
}

void DataFrame::readQML(std::string_view qmlPath) {
    // TODO: open binary file, read header, call quaml::Decode on each column
    (void)qmlPath;
}

void DataFrame::saveQML(std::string_view outPath) {
    // TODO: call quaml::Encode on each column, write header + bitstreams to binary file
    (void)outPath;
}

// Returns the index of ts via binary search, or throws if not found.
static size_t exactIndex(const std::vector<int64_t>& ts, int64_t target) {
    auto it = std::lower_bound(ts.begin(), ts.end(), target);
    if (it == ts.end() || *it != target)
        throw std::out_of_range("getBarExact: timestamp not found");
    return static_cast<size_t>(it - ts.begin());
}

BarView DataFrame::getNextBar(BarView& thisBar){
    //given a bar, get the next bar
    if(!hasNextBar(thisBar)) {
        throw std::out_of_range("getNextBar: next bar not found");
    }

    return BarView{*this, thisBar.idx + 1};
}

bool DataFrame::hasNextBar(BarView& thisBar) const{
    return thisBar.idx < timestamps.size() - 1;
}

BarView DataFrame::getBarExact(int64_t ts) {
    return BarView{*this, exactIndex(timestamps, ts)};
}

BarView DataFrame::getBarInexact(int64_t ts) {
    return getBarInexact(ts, NearestBarType::Nearest);
}

BarView DataFrame::getBarInexact(int64_t ts, NearestBarType type) {
    if (timestamps.empty())
        throw std::out_of_range("getBarInexact: DataFrame is empty");

    auto it = std::lower_bound(timestamps.begin(), timestamps.end(), ts);

    if (type == NearestBarType::Forward) {
        // First timestamp >= ts.
        if (it == timestamps.end())
            throw std::out_of_range("getBarInexact: no bar at or after timestamp");
        return BarView{*this, static_cast<size_t>(it - timestamps.begin())};
    }

    if (type == NearestBarType::Backward) {
        // Last timestamp <= ts.
        if (it == timestamps.begin())
            throw std::out_of_range("getBarInexact: no bar at or before timestamp");
        --it;
        return BarView{*this, static_cast<size_t>(it - timestamps.begin())};
    }

    // NearestBarType::Nearest: pick whichever of the two neighbours is closer.
    if (it == timestamps.end())
        return BarView{*this, timestamps.size() - 1};
    if (it == timestamps.begin())
        return BarView{*this, 0};

    size_t after  = static_cast<size_t>(it - timestamps.begin());
    size_t before = after - 1;
    int64_t distAfter  = timestamps[after]  - ts;
    int64_t distBefore = ts - timestamps[before];
    return BarView{*this, distBefore <= distAfter ? before : after};
}

std::vector<BarView> DataFrame::getBars(int64_t begin, int64_t end) {
    // lower_bound for begin, upper_bound for end gives [begin, end] inclusive.
    auto lo = std::lower_bound(timestamps.begin(), timestamps.end(), begin);
    auto hi = std::upper_bound(timestamps.begin(), timestamps.end(), end);

    std::vector<BarView> result;
    result.reserve(static_cast<size_t>(hi - lo));
    for (auto it = lo; it != hi; ++it)
        result.push_back(BarView{*this, static_cast<size_t>(it - timestamps.begin())});
    return result;
}
