#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

#define BOUND_X 2
#define BOUND_Y 2

#define NTHREADS_X 32
#define NTHREADS_Y 32

#define MIN(a, b) ((a) < (b)) ? (a) : (b)

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__
void sobelKernel(unsigned char* s, unsigned char* t, int *mask, unsigned H, unsigned W, unsigned C) {
    __shared__ int mask_sm[MASK_N * MASK_X * MASK_Y];

    int map_idx = NTHREADS_X * threadIdx.x + threadIdx.y;
    if (map_idx < MASK_N * MASK_X * MASK_Y) {
        mask_sm[map_idx] = mask[map_idx];
    }
    __syncthreads();

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    double val, totalCh;
    int Ch, mask_idx;

    if (x >= 0 && x < W && y >= 0 && y < H) {
        for (int ch = 0; ch < C; ++ch) {
            totalCh = 0.0f;

            for (int ax = 0; ax < MASK_N; ++ax) {
                val = 0.0f;

                for (int v = -BOUND_Y; v < BOUND_Y + 1; ++v) {
                    for (int u = -BOUND_X; u < BOUND_X + 1; ++u) {
                        if ((x + u) >= 0 && (x + u) < W && (y + v) >= 0 && (y + v) < H) {
                            Ch = s[C * (W * (y + v) + (x + u)) + ch];
                            mask_idx = ax * MASK_X * MASK_Y + (u + BOUND_X) * MASK_Y + (v + BOUND_Y);
                            val += Ch * mask_sm[mask_idx];
                        }
                    }
                }

                totalCh += val * val;
            }

            totalCh = sqrt(totalCh) / SCALE;
            t[C * (W * y + x) + ch] = (unsigned char)MIN(totalCh, 255);

        }
    }
}

void sobel(unsigned char* s, unsigned char* t, unsigned H, unsigned W, unsigned C, int img_size) {
    int mask[MASK_N][MASK_X][MASK_Y] = {
        {{ -1, -4, -6, -4, -1},
         { -2, -8,-12, -8, -2},
         {  0,  0,  0,  0,  0},
         {  2,  8, 12,  8,  2},
         {  1,  4,  6,  4,  1}},
        {{ -1, -2,  0,  2,  1},
         { -4, -8,  0,  8,  4},
         { -6,-12,  0, 12,  6},
         { -4, -8,  0,  8,  4},
         { -1, -2,  0,  2,  1}}
    };

    int *mask_d;
    int mask_size = MASK_N * MASK_X * MASK_Y * sizeof(int);
    cudaMalloc(&mask_d, mask_size);
    cudaMemcpy(mask_d, mask, mask_size, cudaMemcpyHostToDevice);

    unsigned char *sd;
    cudaMalloc(&sd, img_size);
    cudaMemcpy(sd, s, img_size, cudaMemcpyHostToDevice);

    unsigned char *td;
    cudaMalloc(&td, img_size);

    dim3 num_blocks((W + NTHREADS_X - 1) / NTHREADS_X, (H + NTHREADS_Y - 1) / NTHREADS_Y, 1);
    dim3 num_threads(NTHREADS_X, NTHREADS_Y, 1);

    sobelKernel<<<num_blocks, num_threads>>>(sd, td, mask_d, H, W, C);

    cudaDeviceSynchronize();
    cudaMemcpy(t, td, img_size, cudaMemcpyDeviceToHost);

    cudaFree(mask_d);
    cudaFree(sd);
    cudaFree(td);
}

int main(int argc, char** argv) {
    assert(argc == 3);

    unsigned height, width, channels;
    unsigned char *src_img, *dst_img;

    read_png(argv[1], &src_img, &height, &width, &channels);
    assert(channels == 3);

    int img_size = height * width * channels * sizeof(unsigned char);
    cudaMallocHost(&dst_img, img_size);

    sobel(src_img, dst_img, height, width, channels, img_size);

    write_png(argv[2], dst_img, height, width, channels);

    free(src_img);
    cudaFree(dst_img);

    return 0;
}
