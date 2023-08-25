#include <iostream>

#include "Graph.h"
//#include "Coordinate.h"
#include "Ordering.h"
#include "Filter.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define INFO
#define VERBOSE
#define DEBUG

int main(int argc, char **argv) {
  // argv[]
  // 0. main
  // 1. query graph
  // 2. data graph
  // 3. output file

  if (argc != 4) {
    std::cerr << "Invalid arguments!" << std::endl;
    return -1;
  }

#ifdef INFO
  std::cout << "------ Sub Graph Isomorphism System ------" << std::endl;
#endif

  //////////////////gpu set/////////////////////////////
  int device = 0;
  cudaSetDevice(device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Device: " << prop.name << std::endl;
#ifdef INFO
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Max threads per block dimension: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " "
            << prop.maxThreadsDim[2] << std::endl;
  std::cout << "Max grid size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2]
            << std::endl;
  std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << std::endl;
  std::cout << "Shared memory per multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
  std::cout << "Total global memory: " << prop.totalGlobalMem << std::endl;
  std::cout << "Total constant memory: " << prop.totalConstMem << std::endl;
  std::cout << "Warp size: " << prop.warpSize << std::endl;
#endif

  //////////////////////load graph////////////////////////
  Graph *g_cpu, *g_gpu, *q_cpu, *q_gpu;
  g_cpu = new Graph();
  q_cpu = new Graph();

#ifdef DEBUG
  puts("1");
#endif

  cudaStream_t loadStream[2];
  for (auto &i: loadStream) cudaStreamCreate(&i);
  cudaMallocAsync((void **) &g_gpu, sizeof(Graph), loadStream[0]);
  cudaMallocAsync((void **) &q_gpu, sizeof(Graph), loadStream[1]);

#ifdef DEBUG
  puts("2");
#endif

  std::vector<std::vector<int>> g_adj, q_adj;

  g_cpu->load(argv[2], g_adj);
  q_cpu->load(argv[1], q_adj);
  q_cpu->candidateSize = g_cpu->nodesCount;
  q_cpu->candidates = (int *) malloc(q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int));
  q_cpu->encoding = (int *) malloc(q_cpu->nodesCount * g_cpu->edgesCount * sizeof(int));
  g_cpu->encoding = (int *) malloc(g_cpu->nodesCount * g_cpu->edgesCount * sizeof(int));

  for (auto &i: loadStream) cudaStreamSynchronize(i);

#ifdef DEBUG
  puts("3");
#endif

#ifdef INFO
  std::cout << "graph loaded" << std::endl;
#endif

// set partitionSize
  g_cpu->partitionSize = 3;
  q_cpu->partitionSize = 3;

////////////////////construct////////////////////////////
  dim3 block(128, 1, 1);
  dim3 grid((g_cpu->nodesCount + block.x - 1) / block.x, 1, 1);

#ifdef DEBUG
  puts("4");
#endif

  g_cpu->construct(g_adj);
  q_cpu->construct(q_adj);

#ifdef INFO
  std::cout << "graph Level-CSR constructed" << std::endl;
#endif

  // Memcpy from cpu to gpu
  cudaStream_t streams[2];
  for (auto &i: streams) cudaStreamCreate(&i);
  cudaMemcpyAsync(g_gpu, g_cpu, sizeof(Graph), cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(q_gpu, q_cpu, sizeof(Graph), cudaMemcpyHostToDevice, streams[1]);
  for (auto &i: streams) cudaStreamSynchronize(i);

#ifdef INFO
  std::cout << "graph copied to GPU" << std::endl;
#endif

  ///////////////Filter//////////////////////
//  int *g_labels_d, *q_labels_d;
//  cudaMalloc((void **) &g_labels_d, g_cpu->nodesCount * sizeof(int));
//  cudaMalloc((void **) &q_labels_d, q_cpu->nodesCount * sizeof(int));

  getEncoding<<<grid, block, 0, streams[0]>>>(g_gpu);
  getEncoding<<<grid, block, 0, streams[1]>>>(q_gpu);
  cudaDeviceSynchronize();

#ifdef INFO
  std::cout << "graph encoding done." << std::endl;
#endif

  firstFiltering<<<grid, block>>>(g_gpu, q_gpu);
  cudaDeviceSynchronize();
  cudaMemcpy(q_cpu, q_gpu, sizeof(Graph), cudaMemcpyDeviceToHost);


#ifdef INFO
  std::cout << "filtering done." << std::endl;
#endif

  /////////////////Ordering///////////////////////////////

  // core-forest decomposition
  getCoreForest(*q_cpu, q_cpu->core, q_cpu->forest);

  for (int i = 0; i < q_cpu->nodesCount; ++i) std::cout << q_cpu->core[i] << ' ';
  std::cout << std::endl;
  for (int i = 0; i < q_cpu->nodesCount; ++i) std::cout << q_cpu->forest[i] << ' ';
  std::cout << std::endl;

//  auto str = cudaGetErrorString(
//    cudaMemcpyAsync(q_gpu->core, q_cpu->core, (q_cpu->nodesCount + 1) * sizeof(int), cudaMemcpyHostToDevice,
//                    streams[0]));
//  std::cout << str << std::endl;
//  std::cout << "a" << std::endl;
//  cudaGetErrorString(
//    cudaMemcpyAsync(q_gpu->forest, q_cpu->forest, (q_cpu->nodesCount + 1) * sizeof(int), cudaMemcpyHostToDevice,
//                    streams[1]));
//  std::cout << str << std::endl;
//  std::cout << "B" << std::endl;

  cudaMemcpyAsync(q_gpu, q_cpu, sizeof(Graph), cudaMemcpyHostToDevice, streams[0]);
//  for (auto &i: streams) cudaStreamSynchronize(i);

#ifdef INFO
  std::cout << "core-forest decomposition done." << std::endl;
#endif
  // get score - determine the order of join
  getScore<<<grid, block>>>(q_gpu);
  cudaDeviceSynchronize();
  std::cout << "OK" << std::endl;

#ifdef DEBUG
  cudaMemcpy(q_cpu, q_gpu, sizeof(Graph), cudaMemcpyDeviceToHost);
  for (int i = 0; i < q_cpu->nodesCount; ++i) std::cout << q_cpu->score[i] << ' ';
  std::cout << std::endl;
#endif


#ifdef INFO
  std::cout << "score done." << std::endl;
#endif

  std::vector<std::pair<int, int>> score_pairs;
  for (int i = 0; i < q_cpu->nodesCount; ++i) score_pairs.emplace_back(q_cpu->score[i], i);
  std::sort(score_pairs.begin(), score_pairs.end(), std::greater<>());
  for(int i = 0;i<q_cpu->nodesCount;++i) q_cpu->order[i] = score_pairs[i].second;
  /////////////////Join///////////////////////////////

  cudaDeviceReset();
  cudaDeviceSynchronize();

  return 0;
}
