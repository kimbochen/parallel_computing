//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "sha256.h"

#define NTHREADS 256
#define NBLOCKS 16777215

////////////////////////   Device Variable   /////////////////////

__device__ unsigned int found_sol;


////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;


////////////////////////   Utils   ///////////////////////

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }

    return '\0';
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(; s < string_len; s += 2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i = 0; i < len; ++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
__host__ __device__
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i = len - 1; i >= 0; --i)
    {
        printf("%02x", hex[i]);
    }
}

__device__
int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i = byte_len - 1; i >= 0; --i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Nonce Hash   ///////////////////////

__global__
void double_sha256_nonce(SHA256 *sha256_ctx, HashBlock *block, unsigned char *target_hex)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (!found_sol && tid < 0xffffffff)
    {
        SHA256 sha256_ctx_local, tmp;
        HashBlock block_local;
        BYTE *bytes;

        memcpy(&block_local, block, sizeof(HashBlock));
        memcpy(&sha256_ctx_local, sha256_ctx, sizeof(SHA256));

        bytes = (unsigned char *)&block_local;
        block_local.nonce = tid;

        sha256(&tmp, bytes, sizeof(HashBlock));
        sha256(&sha256_ctx_local, (BYTE*)&tmp, sizeof(SHA256));

        // if(block_local.nonce % 100000 == 0)
        // {
        //     printf("hash #%10u (big): ", block_local.nonce);
        //     print_hex_inverse(sha256_ctx_local.b, 32);
        //     printf("\n");
        // }

        if(little_endian_bit_comparison(sha256_ctx_local.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
        {
            printf("Found Solution!!\n");
            printf("hash #%10u (big): ", block_local.nonce);
            print_hex_inverse(sha256_ctx_local.b, 32);
            printf("\n\n");

            unsigned int old = atomicCAS(&found_sol, 0, 1);

            memcpy(block, &block_local, sizeof(HashBlock));
            memcpy(sha256_ctx, &sha256_ctx_local, sizeof(SHA256));
        }
    }
}

////////////////////////  Merkle Root Hash   ///////////////////////

void double_sha256_merkle(unsigned char *list, size_t total_count)
{
    SHA256 tmp;
    size_t sha256_size = sizeof(SHA256);

    if (total_count % 64 == 32)
    {
        memcpy(&list[total_count], &list[total_count - 32], 32);
    }

    for(int i = 0, j = 0; i < total_count; i += 64, j += 32)
    {
        sha256(&tmp, (BYTE*)&list[i], 64);  // tmp = hash(list[i]+list[i + 1])
        sha256((SHA256*)&list[j], (BYTE*)&tmp, sha256_size);  // list[j] = hash(tmp)
    }
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *list = new unsigned char[(total_count + 1) * 32];

    // copy each branch to the list
    for(int i = 0; i < total_count; ++i)
    {
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(&list[i * 32], branch[i], 64);
    }

    // calculate merkle root
    for (; total_count > 1; total_count = (total_count + 1) / 2)
    {
        double_sha256_merkle(list, total_count * 32);
    }

    memcpy(root, &list[0], 32);

    delete[] list;
}


void solve(FILE *fin, FILE *fout)
{

    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i = 0; i < tx; ++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");


    // **** solve block ****
    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
    block.nonce = 0;
    
    
    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    
    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");


    // ********** find nonce **************
    
    SHA256 sha256_ctx;
    size_t sha256_size = sizeof(SHA256);
    size_t hashblock_size = sizeof(HashBlock);

    SHA256 *sha256_ctx_dev;
    cudaMalloc(&sha256_ctx_dev, sha256_size);
    cudaMemcpy(sha256_ctx_dev, &sha256_ctx, sha256_size, cudaMemcpyHostToDevice);

    HashBlock *block_dev;
    cudaMalloc(&block_dev, hashblock_size);
    cudaMemcpy(block_dev, &block, hashblock_size, cudaMemcpyHostToDevice);

    unsigned char *target_hex_dev;
    cudaMalloc(&target_hex_dev, 32);
    cudaMemcpy(target_hex_dev, target_hex, 32, cudaMemcpyHostToDevice);

    const unsigned int zero = 0;
    cudaMemcpyToSymbol(found_sol, &zero, sizeof(unsigned int));

    double_sha256_nonce<<<NBLOCKS, NTHREADS>>>(sha256_ctx_dev, block_dev, target_hex_dev);
    cudaDeviceSynchronize();

    cudaMemcpy(&sha256_ctx, sha256_ctx_dev, sha256_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&block, block_dev, hashblock_size, cudaMemcpyDeviceToHost);

    // print result

    //little-endian
    printf("hash(little): ");
    print_hex(sha256_ctx.b, 32);
    printf("\n");

    //big-endian
    printf("hash(big):    ");
    print_hex_inverse(sha256_ctx.b, 32);
    printf("\n\n");

    for(int i = 0; i < 4; ++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;

    cudaFree(sha256_ctx_dev);
    cudaFree(block_dev);
    cudaFree(target_hex_dev);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for(int i = 0; i < totalblock; ++i)
    {
        solve(fin, fout);
    }

    return 0;
}
