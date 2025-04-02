#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initialize_grid(float *T, int N, float T_top, float T_other) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == 0)
        T[i * N + j] = T_top; // Fixed top boundary temperature
      else
        T[i * N + j] = T_other; // Remaining grid starts at T_other
    }
  }
}

void heat_diffusion_step(float *T, float *T_new, int N, int boundary_row,
                         float alpha1, float alpha2) {
#pragma omp parallel for
  for (int i = 1; i < N - 1; i++) {
#pragma omp simd
    for (int j = 1; j < N - 1; j++) {
      float alpha = (i < boundary_row) ? alpha1 : alpha2;
      T_new[i * N + j] =
          (1 - alpha) * T[i * N + j] +
          (alpha / 4.0f) * (T[(i + 1) * N + j] + T[(i - 1) * N + j] +
                            T[i * N + (j + 1)] + T[i * N + (j - 1)]);
    }
  }
}

void save_grid_to_file(float *T, int N, const char *filename) {
  FILE *file = fopen(filename, "w");
  if (!file) {
    perror("Error opening file for writing");
    return;
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      fprintf(file, "%.2f", T[i * N + j]);
      if (j < N - 1)
        fprintf(file, ",");
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

// Function to get the time difference in seconds
double get_time_diff(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char *argv[]) {
  if (argc < 8) {
    printf("Usage: %s <grid_size> <boundary_row> <alpha1> <alpha2> "
           "<iterations> <T_top> <T_other>\n",
           argv[0]);
    return 1;
  }

  int N = atoi(argv[1]);
  int boundary_row = atoi(argv[2]);
  float alpha1 = atof(argv[3]);
  float alpha2 = atof(argv[4]);
  int iterations = atoi(argv[5]);
  float T_top = atof(argv[6]);
  float T_other = atof(argv[7]);

  if (N < 10 || boundary_row < 1 || boundary_row >= N - 1 || alpha1 < 0.0f ||
      alpha1 > 1.0f || alpha2 < 0.0f || alpha2 > 1.0f || iterations < 1) {
    printf("Error: Invalid parameters. Ensure:\n");
    printf("  - grid_size >= 10\n  - 1 <= boundary_row < grid_size - 1\n");
    printf("  - 0.0 <= alpha1, alpha2 <= 1.0\n  - iterations >= 1\n");
    return 1;
  }

  float *T = (float *)malloc(N * N * sizeof(float));
  float *T_new = (float *)malloc(N * N * sizeof(float));

  initialize_grid(T, N, T_top, T_other);

  // Start timing with high precision
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int j = 0; j < N; j++) {
    T_new[j] = T_top;
  }

  for (int iter = 0; iter < iterations; iter++) {
    heat_diffusion_step(T, T_new, N, boundary_row, alpha1, alpha2);
    float *temp = T;
    T = T_new;
    T_new = temp;
  }

  // End timing
  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed_time = get_time_diff(start, end);

  save_grid_to_file(T, N, "heat_output.csv");

  free(T);
  free(T_new);

  printf("Simulation complete. Results saved to heat_output.csv\n");
  printf("Time taken for %d iterations: %.6f seconds\n", iterations,
         elapsed_time);
  return 0;
}
