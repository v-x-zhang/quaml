#pragma once

#include <cstdint>
#include <queue>
#include <vector>
#include <unordered_map>

class Huffman {
private:
    struct TreeNode {
        int32_t freq;
        int32_t symbol;
        bool is_leaf;

        TreeNode* left;
        TreeNode* right;

        TreeNode() : freq(0), symbol(0), is_leaf(false), left(nullptr), right(nullptr) {}
    };

    struct CompareNodes {
        bool operator()(const TreeNode* lhs, const TreeNode* rhs) {
            return lhs->freq > rhs->freq;
        }
    };

    TreeNode* head;
    std::vector<TreeNode*> allNodes;
    std::vector<int32_t> symbols;
    std::unordered_map<int32_t, std::string> map;

    void BuildHelper(TreeNode*, const std::string&, std::unordered_map<int32_t, std::string>&);

public:
    Huffman();
    ~Huffman();

    void BuildTree(const std::vector<int32_t>& input);
    void BuildMap();

    std::string Encode();
    std::vector<int32_t> Decode(const std::string& bits);

    const std::unordered_map<int32_t, std::string>& GetMap() const { return map; }
};
