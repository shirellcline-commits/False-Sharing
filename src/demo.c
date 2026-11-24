#include <stdio.h>
#include <omp.h>

#define N 200000000
#define HILOS 4

int main() {
    double t0, t_seq, t_fs, t_pad, t_loc;

    // 1. SECUENCIAL (Referencia base)
    int seq[4] = {0};
    t0 = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        unsigned int x = i;
        x = x * 1103515245 + 12345;
        int bin = (x / 65536) % 4;
        seq[bin]++;
    }
    t_seq = omp_get_wtime() - t0;

    // 2. FALSE SHARING: 4 hilos x 4 ints = 64 bytes (todos en 1 misma línea de caché)
    int fs[HILOS][4] = {0};
    t0 = omp_get_wtime();
    #pragma omp parallel num_threads(HILOS)
    {
        int id = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < N; i++) {
            unsigned int x = i;
            x = x * 1103515245 + 12345;
            int bin = (x / 65536) % 4;
            fs[id][bin]++; // Invalidación constante de caché entre núcleos
        }
    }
    t_fs = omp_get_wtime() - t0;

    // 3. PADDING: 16 ints por fila = 64 bytes por hilo (4 útiles + 12 de relleno)
    int pad[HILOS][16] __attribute__((aligned(64))) = {0};
    t0 = omp_get_wtime();
    #pragma omp parallel num_threads(HILOS)
    {
        int id = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < N; i++) {
            unsigned int x = i;
            x = x * 1103515245 + 12345;
            int bin = (x / 65536) % 4;
            pad[id][bin]++; // Cada hilo escribe en su propia línea de caché
        }
    }
    t_pad = omp_get_wtime() - t0;

    // 4. VARIABLES LOCALES (Privado + combinar)
    int total[4] = {0};
    t0 = omp_get_wtime();
    #pragma omp parallel num_threads(HILOS)
    {
        int local[4] = {0}; // Vive en el stack privado de cada hilo
        #pragma omp for
        for (int i = 0; i < N; i++) {
            unsigned int x = i;
            x = x * 1103515245 + 12345;
            int bin = (x / 65536) % 4;
            local[bin]++;
        }
        #pragma omp critical
        for (int b = 0; b < 4; b++) total[b] += local[b];
    }
    t_loc = omp_get_wtime() - t0;

    // Verificación de que el conteo es exacto en las 4 versiones
    printf("Conteo cubeta 0 -> Seq: %d | FS: %d | Pad: %d | Local: %d\n\n",
           seq[0],
           fs[0][0] + fs[1][0] + fs[2][0] + fs[3][0],
           pad[0][0] + pad[1][0] + pad[2][0] + pad[3][0],
           total[0]);

    printf("1. Secuencial:    %.4f s (Speedup: 1.00x)\n", t_seq);
    printf("2. False Sharing: %.4f s (Speedup: %.2fx)\n", t_fs,  t_seq / t_fs);
    printf("3. Con Padding:   %.4f s (Speedup: %.2fx)\n", t_pad, t_seq / t_pad);
    printf("4. Vars Locales:  %.4f s (Speedup: %.2fx)\n", t_loc, t_seq / t_loc);

    return 0;
}