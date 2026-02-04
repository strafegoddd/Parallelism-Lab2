#include <complex>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <x86intrin.h>
#include <cstddef>


using namespace std;


void transposeAVX2(double* out, const double* in, size_t rows, size_t cols)
{
    const size_t blockSize = 4; // 4 double = 256 бит

    size_t i, j;
    for (i = 0; i + blockSize <= rows; i += blockSize) {
        for (j = 0; j + blockSize <= cols; j += blockSize) {

            // Загружаем блок 4×4
            __m256d row0 = _mm256_loadu_pd(&in[(i+0)*cols + j]); // a0 a1 a2 a3
            __m256d row1 = _mm256_loadu_pd(&in[(i+1)*cols + j]); // b0 b1 b2 b3
            __m256d row2 = _mm256_loadu_pd(&in[(i+2)*cols + j]); // c0 c1 c2 c3
            __m256d row3 = _mm256_loadu_pd(&in[(i+3)*cols + j]); // d0 d1 d2 d3

            // Разбиваем и переставляем пары строк
            __m256d t0 = _mm256_unpacklo_pd(row0, row1); // a0 b0 a1 b1
            __m256d t1 = _mm256_unpackhi_pd(row0, row1); // a2 b2 a3 b3
            __m256d t2 = _mm256_unpacklo_pd(row2, row3); // c0 d0 c1 d1
            __m256d t3 = _mm256_unpackhi_pd(row2, row3); // c2 d2 c3 d3

            // Собираем верх и низ блока
            __m256d r0 = _mm256_permute2f128_pd(t0, t2, 0x20); // a0 b0 c0 d0
            __m256d r1 = _mm256_permute2f128_pd(t1, t3, 0x20); // a2 b2 c2 d2
            __m256d r2 = _mm256_permute2f128_pd(t0, t2, 0x31); // a1 b1 c1 d1
            __m256d r3 = _mm256_permute2f128_pd(t1, t3, 0x31); // a3 b3 c3 d3

            // Сохраняем транспонированный блок
            _mm256_storeu_pd(&out[(j+0)*rows + i], r0); // a0 b0 c0 d0
            _mm256_storeu_pd(&out[(j+1)*rows + i], r1); // a1 b1 c1 d1
            _mm256_storeu_pd(&out[(j+2)*rows + i], r2); // a2 b2 c2 d2
            _mm256_storeu_pd(&out[(j+3)*rows + i], r3); // a3 b3 c3 d3
        }


        // хвостовые столбцы справа
        for (; j < cols; ++j) {
            for (size_t ii = 0; ii < blockSize && (i+ii)<rows; ++ii)
                out[j*rows + (i+ii)] = in[(i+ii)*cols + j];
        }
    }

    // хвостовые строки снизу
    for (; i < rows; ++i) {
        for (j = 0; j < cols; ++j)
            out[j*rows + i] = in[i*cols + j];
    }
}


void matrixPrint(double *matrix, size_t rows, size_t cols) {
    cout << "Matrix:\n";
    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++)
            cout << matrix[r * cols + c] << ' ';
        cout << endl;
    }
}


void matrixMul(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){
    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            double accum = 0;
            for (size_t i = 0; i < sharedDim; i++)
                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
            result[r1 * colsB + c2] = accum;
        }
}

void matrixMulAVX2(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){

    std::vector<double> TransposeT(colsB * sharedDim, 0.0);

    if (sharedDim % 4 == 0 && colsB % 4 == 0)
        transposeAVX2(TransposeT.data(), matB, colsB, sharedDim);
    else {
        for (size_t r = 0; r < sharedDim; ++r)
            for (size_t c = 0; c < colsB; ++c)
                TransposeT[c * sharedDim + r] = matB[r * colsB + c];
    }

    const double *matT = TransposeT.data();
    double temp[4];

    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            __m256d sum_vec = _mm256_setzero_pd();
            size_t k = 0;

            // Умножаем блоками по 4 double
            for (; k + 3 < sharedDim; k += 4) {
                __m256d x = _mm256_loadu_pd(&matA[r1 * sharedDim + k]);
                __m256d y = _mm256_loadu_pd(&matT[c2 * sharedDim + k]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(x, y));
            }

            // Суммируем 4 элемента внутри регистра
            _mm256_storeu_pd(temp, sum_vec);
            double result_val = temp[0] + temp[1] + temp[2] + temp[3];

            // Хвостовые элементы
            for (; k < sharedDim; ++k) {
                result_val += matA[r1 * sharedDim + k] * matT[c2 * sharedDim + k];
            }

            result[r1 * colsB + c2] = result_val;
        }
}


void matrixMulAVX2_gather(double *result, const double *matA, const double *matB,
                          size_t sharedDim, size_t rowsA, size_t colsB) {

    double temp[4];
    for (size_t r1 = 0; r1 < rowsA; r1++) {
        for (size_t c2 = 0; c2 < colsB; c2++) {
            __m256d sum_vec = _mm256_setzero_pd();
            size_t k = 0;

            // Умножаем блоками по 4 double
            for (; k + 3 < sharedDim; k += 4) {
                __m256d a_vec = _mm256_loadu_pd(&matA[r1 * sharedDim + k]);

                __m256i indices = _mm256_setr_epi64x(
                        k * colsB + c2,
                        (k + 1) * colsB + c2,
                        (k + 2) * colsB + c2,
                        (k + 3) * colsB + c2
                );
                __m256d b_vec = _mm256_i64gather_pd(matB, indices, sizeof(double));

                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }

            // Суммируем 4 элемента внутри регистра
            _mm256_storeu_pd(temp, sum_vec);
            double result_val = temp[0] + temp[1] + temp[2] + temp[3];

            // Хвостовые элементы
            for (; k < sharedDim; ++k) {
                result_val += matA[r1 * sharedDim + k] * matB[k * colsB + c2];
            }

            result[r1 * colsB + c2] = result_val;
        }
    }
}

//void matrixMul_clear(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){
//    for (size_t r1 = 0; r1 < rowsA / sizeof(__m256d); r1++)
//        for (size_t c2 = 0; c2 < colsB; c2++) {
//            __m256d accum = _mm256_setzero_pd();
//            for (size_t i = 0; i < sharedDim; i++)
//                __m256d x = _mm256_loadu_pd(&matA[r1 * sharedDim + i]);
//                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
//            result[r1 * colsB + c2] = accum;
//        }
//}



void speedtest(size_t rowsA, size_t sharedDim, size_t colsB) {
    std::vector<double> A = std::vector<double>(rowsA * sharedDim, 1.0);
    std::vector<double> B = std::vector<double>(colsB * sharedDim);
    std::vector<double> R1 = std::vector<double>(rowsA * colsB, 0.0);
    std::vector<double> R2 = std::vector<double>(rowsA * colsB, 0.0);
    std::vector<double> R3 = std::vector<double>(rowsA * colsB, 0.0);

    for (size_t i = 0; i < colsB * sharedDim; ++i) {
        B[i] = (double) i;
    }

    auto t1 = std::chrono::steady_clock::now();
    matrixMul(R1.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    auto t2 = std::chrono::steady_clock::now();
//    matrixPrint(R1.data(), rowsA, colsB);

    auto t3 = std::chrono::steady_clock::now();
    matrixMulAVX2(R2.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    auto t4 = std::chrono::steady_clock::now();
//    matrixPrint(R2.data(), rowsA, colsB);

    auto t5 = std::chrono::steady_clock::now();
    matrixMulAVX2_gather(R3.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    auto t6 = std::chrono::steady_clock::now();
//    matrixPrint(R3.data(), rowsA, colsB);

    cout << "Duration of synchronous calc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;
    cout << "Duration of AVX2 calc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << endl;
    cout << "Duration of AVX2_Gather calc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << endl;
}


void speedtest_avg_openmp(size_t n) {
    std::cout << "======= SPEEDTEST MATRIX MUL ==========" << std::endl;

    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output.csv"));
    if (!output.is_open())
    {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    output << "Sync,AVX2,AVX2Acceleration,Gather,GatherAcceleration\n";

    std::vector<double> A = std::vector<double>(n * n, 1.0);
    std::vector<double> B = std::vector<double>(n * n);
    std::vector<double> R1 = std::vector<double>(n * n, 0.0);
    std::vector<double> R2 = std::vector<double>(n * n, 0.0);
    std::vector<double> R3 = std::vector<double>(n * n, 0.0);

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (double) i;
    }

    auto t1 = std::chrono::steady_clock::now();
    matrixMul(R1.data(), A.data(), B.data(), n, n, n);
    auto t2 = std::chrono::steady_clock::now();

    auto t3 = std::chrono::steady_clock::now();
    matrixMulAVX2(R2.data(), A.data(), B.data(), n, n, n);
    auto t4 = std::chrono::steady_clock::now();

    auto t5 = std::chrono::steady_clock::now();
    matrixMulAVX2_gather(R3.data(), A.data(), B.data(), n, n, n);
    auto t6 = std::chrono::steady_clock::now();

    std::cout << "Sync time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << "\t| AVX2 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
              << "\t| AVX2 Acceleration: " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
              << "\t| Gather time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count()
              << "\t| Gather Acceleration: " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count()
              << endl;

    output << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << ","
           << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << ","
           << (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << ","
           << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << ","
           << (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count()
           << endl;

    output.close();
}


int main() {
    cout << "Test 1:" << endl;
    std::size_t rowsA = 2, sharedDim = 3, colsB = 2;
    std::vector<double> A(rowsA * sharedDim, 1.0);
    std::vector<double> B(sharedDim * colsB, 2.0);
    std::vector<double> R(rowsA * colsB, 0.0);

    matrixMul(R.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    matrixPrint(R.data(), rowsA, colsB);

    cout << "Test 2:" << endl;
    rowsA = 2; sharedDim = 3; colsB = 2;
    A = std::vector<double>({ 1.0, 2.0, -3.0,
                              -2.0, 13.0, -2.0});
    B = std::vector<double>({3.0, 4.0,
                             5.0, -1.0,
                             4.0, 4.0});
    R = std::vector<double>(rowsA * colsB, 0.0);

    matrixMul(R.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    matrixPrint(R.data(), rowsA, colsB);

    speedtest_avg_openmp(1000);
    return 0;
}
