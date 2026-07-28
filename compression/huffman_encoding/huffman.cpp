#include "huffman.h"

Huffman::Huffman() : head(nullptr) {}

void Huffman::BuildTree(const std::vector<int32_t>& input) {
    symbols = input;

    std::unordered_map<int32_t, TreeNode*> freqs;
    std::priority_queue<TreeNode*, std::vector<TreeNode*>, CompareNodes> heap;

    for (int32_t s : input) {
        if (freqs.find(s) == freqs.end()) {
            TreeNode* node = new TreeNode();
            node->symbol = s;
            node->freq = 1;
            node->is_leaf = true;
            freqs[s] = node;
        } else {
            freqs[s]->freq++;
        }
    }

    for (auto& [sym, node] : freqs) {
        heap.push(node);
        allNodes.push_back(node);
    }

    // Single unique symbol: wrap in a dummy root so encoding still works.
    if (heap.size() == 1) {
        TreeNode* only = heap.top(); heap.pop();
        TreeNode* root = new TreeNode();
        root->freq = only->freq;
        root->left = only;
        allNodes.push_back(root);
        heap.push(root);
    }

    while (heap.size() > 1) {
        TreeNode* a = heap.top(); heap.pop();
        TreeNode* b = heap.top(); heap.pop();

        TreeNode* merged = new TreeNode();
        merged->freq = a->freq + b->freq;
        merged->left = a;
        merged->right = b;

        allNodes.push_back(merged);
        heap.push(merged);
    }

    head = heap.top();
}

void Huffman::BuildMap() {
    BuildHelper(head, "", map);
}

void Huffman::BuildHelper(TreeNode* node, const std::string& bits,
                           std::unordered_map<int32_t, std::string>& m) {
    if (node->is_leaf) {
        m[node->symbol] = bits;
        return;
    }
    if (node->left)  BuildHelper(node->left,  bits + "0", m);
    if (node->right) BuildHelper(node->right, bits + "1", m);
}

std::string Huffman::Encode() {
    std::string out;
    for (int32_t s : symbols) {
        out += map.at(s);
    }
    return out;
}

std::vector<int32_t> Huffman::Decode(const std::string& bits) {
    std::vector<int32_t> out;
    TreeNode* cur = head;
    for (char b : bits) {
        cur = (b == '0') ? cur->left : cur->right;
        if (cur->is_leaf) {
            out.push_back(cur->symbol);
            cur = head;
        }
    }
    return out;
}

Huffman::~Huffman() {
    for (int i = (int)allNodes.size() - 1; i >= 0; i--) {
        delete allNodes[i];
    }
}
