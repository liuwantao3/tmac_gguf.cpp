#include <cstdio>
#include <cstdint>
#include <cstring>

struct Tensor {
    char name[128];
    uint64_t rows, cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
};

constexpr int TENSOR_F32 = 0;
constexpr int TENSOR_F16 = 1;
constexpr int TENSOR_Q8_0 = 8;
constexpr int TENSOR_Q6_K = 14;
constexpr int TENSOR_Q5_0 = 6;
constexpr int TENSOR_Q4_K = 12;

const char* type_name(int t) {
    switch (t) {
        case TENSOR_F32: return "F32";
        case TENSOR_F16: return "F16";
        case TENSOR_Q8_0: return "Q8_0";
        case TENSOR_Q6_K: return "Q6_K";
        case TENSOR_Q5_0: return "Q5_0";
        case TENSOR_Q4_K: return "Q4_K";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 4, SEEK_SET); // skip magic
    uint64_t n;
    fread(&n, 8, 1, f);
    printf("Total tensors: %llu\n", (unsigned long long)n);
    int counts[20] = {0};
    for (uint64_t i = 0; i < n; i++) {
        uint64_t nl, rows, cols, nb;
        uint32_t type;
        char name[128];
        fread(&nl, 8, 1, f);
        if (nl >= 128) { fread(name, 1, 127, f); name[127] = 0; fseek(f, nl-127, SEEK_CUR); }
        else { fread(name, 1, nl, f); name[nl] = 0; }
        fread(&rows, 8, 1, f);
        fread(&cols, 8, 1, f);
        fread(&type, 4, 1, f);
        fread(&nb, 8, 1, f);
        fseek(f, nb, SEEK_CUR);
        if (type < 20) counts[type]++;
        // Print attention and FFN tensors
        if (strstr(name, "attn") || strstr(name, "ffn") || strstr(name, "token_embd") || strstr(name, "output_norm")) {
            printf("  %-40s type=%-6s rows=%-6llu cols=%-6llu\n", name, type_name(type),
                   (unsigned long long)rows, (unsigned long long)cols);
        }
    }
    printf("\nType counts:\n");
    for (int i = 0; i < 20; i++) {
        if (counts[i]) printf("  type %2d (%s): %d\n", i, type_name(i), counts[i]);
    }
    fclose(f);
    return 0;
}
