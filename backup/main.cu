#include <iostream>

#include "Graph.h"
//#include "Coordinate.h"
#include "Ordering.h"
#include "Filter.h"
#include "Join.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>

#define INFO
#define VERBOSE
#define DEBUG

int main(int argc, char **argv) {
  // argv[]
  // 0. main
  // 1. query graph
  // 2. data graph
  // 3. output file

  // build/SubGraph ~/code/dataset/Mydata/query.graph ~/code/dataset/Mydata/data.graph ./result.txt
//
//  if (argc != 4) {
//    std::cerr << "Invalid arguments!" << std::endl;
//    return -1;
//  }

  argv = (char **) malloc(4 * sizeof(char *));
  argv[0] = (char *) malloc(10 * sizeof(char));
  argv[1] = (char *) malloc(100 * sizeof(char));
  argv[2] = (char *) malloc(100 * sizeof(char));
  argv[3] = (char *) malloc(100 * sizeof(char));

  strcpy(argv[0], "build/SubGraph");
  strcpy(argv[1], "/home/z/code/dataset/Mydata/query.graph");
  strcpy(argv[2], "/home/z/code/dataset/Mydata/data.graph");
  strcpy(argv[3], "./result.txt");

#ifdef INFO
  std::cout << "------ Sub Graph Isomorphism System ------" << std::endl;
#endif

  //////////////////gpu set/////////////////////////////
  int device = 0;
  cudaSetDevice(device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Device: " << prop.name << std::endl;
  cudaDeviceReset();
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
  auto e = cudaGetErrorString(cudaMallocAsync((void **) &q_gpu, sizeof(Graph), loadStream[1]));
  for (auto &i: loadStream) cudaStreamSynchronize(i);
  std::cout << e << std::endl;

//  std::cout << sizeof(Graph) << std::endl;
#ifdef DEBUG
  puts("2");
#endif

  std::vector<std::vector<std::pair<VID_t, Label_t>>> g_adj, q_adj;

  g_cpu->load(argv[2], g_adj);
  q_cpu->load(argv[1], q_adj);
  q_cpu->candidateSize = g_cpu->nodesCount;
  g_cpu->encodingSize = q_cpu->encodingSize = g_cpu->edgeLabelsCount;
  q_cpu->candidates = (int *) malloc(q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int));
  q_cpu->encoding = (int *) malloc(q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int));
  g_cpu->encoding = (int *) malloc(g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int));

  std::cout << "aaa\n";
//  cudaMalloc((void **) &(q_gpu->encoding), q_cpu->nodesCount * g_cpu->encodingSize * sizeof(int));
//  cudaMalloc((void **) &(g_gpu->encoding), g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int));
//  cudaMalloc((void **) &(q_gpu->candidates), q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int));

  int *q_encoding_device, *g_encoding_device, *q_candidates_device;
  auto err = cudaGetErrorString(cudaMalloc((void **) &q_encoding_device, q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int)));
  std::cout << err << std::endl;
  err = cudaGetErrorString(cudaMalloc((void **) &g_encoding_device, g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int)));
  std::cout << err << std::endl;
  err = cudaGetErrorString(cudaMalloc((void **) &q_candidates_device, q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int)));
  std::cout << err << std::endl;
  cudaMemset(q_candidates_device, 0, q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int));
  cudaMemset(q_encoding_device, 0, q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int));
  cudaMemset(g_encoding_device, 0, g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int));
  std::cout << "BBB\n";
  cudaStream_t memcpyStream[3];
  for(auto &i: memcpyStream) cudaStreamCreate(&i);
  cudaMemcpyAsync(&q_gpu->encoding, &q_encoding_device, q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int), cudaMemcpyDeviceToDevice, memcpyStream[0]);
  cudaMemcpyAsync(&g_gpu->encoding, &g_encoding_device, g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int), cudaMemcpyDeviceToDevice, memcpyStream[1]);
  cudaMemcpyAsync(&q_gpu->candidates, &q_candidates_device, q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int), cudaMemcpyDeviceToDevice, memcpyStream[2]);
  for(auto &i: memcpyStream) cudaStreamSynchronize(i);

//  cudaMemcpy(&(q_gpu[0].encoding), q_encoding_device, q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int), cudaMemcpyDeviceToDevice);
//  cudaMemcpy(&(g_gpu[0].encoding), g_encoding_device, g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int), cudaMemcpyDeviceToDevice);

//
//
//  e = cudaGetErrorString(cudaMemcpy(&(q_gpu->encoding), q_encoding_device, q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int), cudaMemcpyDeviceToDevice));
//  std::cout << e << std::endl;

//  q_gpu->encoding = q_encoding_device;
//  g_gpu->encoding = g_encoding_device;
//  q_gpu->candidates = q_candidates_device;
  std::cout << "bbb\n";

  memset(q_cpu->candidates, 0, q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int));
  memset(q_cpu->encoding, 0, q_cpu->nodesCount * g_cpu->edgesCount * sizeof(int));
  memset(g_cpu->encoding, 0, g_cpu->nodesCount * g_cpu->edgesCount * sizeof(int));






#ifdef DEBUG
  // print graph
  for (int i = 0; i < q_cpu->nodesCount; ++i)
    std::cout << q_cpu->nodes[i] << ' ';
  std::cout << '\n';
  for (int i = 0; i < q_cpu->nodesCount; ++i)
    std::cout << q_cpu->index1[i] << ' ';
  std::cout << '\n';
  for (int i = 0; i < q_cpu->edgesCount; ++i)
    std::cout << q_cpu->index2[i] << ' ';
  std::cout << '\n';
  for (int i = 0; i < q_cpu->edgesCount; ++i)std::cout << q_cpu->column[i] << ' ';
  std::cout << '\n';
  for (int i = 0; i < q_cpu->edgesCount; ++i) std::cout << q_cpu->edges_labels[i] << ' ';
  std::cout << '\n';
  std::cout << "degree\n";
  for (int i = 0; i < q_cpu->nodesCount; ++i) std::cout << q_cpu->getDegree(i) << ' ';
  std::cout << '\n';
#endif

#ifdef DEBUG
  puts("3");
#endif

#ifdef INFO
  std::cout << "graph loaded" << std::endl;
#endif

// set partitionSize
  g_cpu->partitionSize = 32;
  q_cpu->partitionSize = 32;

////////////////////construct////////////////////////////
  dim3 block(128, 1, 1);
  dim3 grid((g_cpu->nodesCount + block.x - 1) / block.x, 1, 1);

#ifdef DEBUG
  puts("4");
#endif

  g_cpu->construct(g_adj);
  std::cout << " --- \n";
  q_cpu->construct(q_adj);

#ifdef INFO
  std::cout << "graph Level-CSR constructed" << std::endl;
#endif

#ifdef DEBUG
  for (int i = 0; i < q_cpu->nodesCount; ++i) std::cout << q_cpu->index1[i] << ' ';
  std::cout << '\n';
  for (int i = 0; i < q_cpu->edgesCount; ++i) std::cout << q_cpu->index2[i] << ' ';
  std::cout << '\n';
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

#ifdef DEBUG
  cudaMemcpyAsync(g_cpu, g_gpu, sizeof(Graph), cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(q_cpu, q_gpu, sizeof(Graph), cudaMemcpyDeviceToHost, streams[1]);
  for (auto &i: streams) cudaStreamSynchronize(i);
  std::cout << "printing encoding of g\n";
  for (int i = 0; i < g_cpu->nodesCount; ++i) {
    for (int j = 0; j < g_cpu->edgesCount; ++j) std::cout << g_cpu->encoding[i * g_cpu->edgesCount + j] << ' ';
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "printing encoding of q\n";
  for (int i = 0; i < q_cpu->nodesCount; ++i) {
    for (int j = 0; j < q_cpu->edgesCount; ++j) std::cout << q_cpu->encoding[i * q_cpu->edgesCount + j] << ' ';
    std::cout << std::endl;
  }

#endif

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

  cudaMemcpy(q_gpu, q_cpu, sizeof(Graph), cudaMemcpyHostToDevice);
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
  for (int i = 0; i < q_cpu->nodesCount; ++i) score_pairs.push_back({q_cpu->score[i], i});
  std::sort(score_pairs.begin(), score_pairs.end(), std::greater<>());
  for (int i = 0; i < q_cpu->nodesCount; ++i) q_cpu->order[i] = score_pairs[i].second;

#ifdef INFO
  std::cout << "done." << std::endl;
#endif
  /////////////////Join///////////////////////////////

  cudaMemcpyAsync(q_cpu, q_gpu, sizeof(Graph), cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(g_cpu, g_gpu, sizeof(Graph), cudaMemcpyDeviceToHost, streams[1]);
  cudaStreamSynchronize(streams[0]);
  cudaStreamSynchronize(streams[1]);

  int *M;
  M = (int *) malloc(q_cpu->nodesCount * sizeof(int));
  bool res = join(q_cpu, g_cpu, M, 0);
  if (!res) std::cout << "No solution." << std::endl, exit(0);

  /////////////////Output///////////////////////////////
  std::fstream fout(argv[3]);
  if (!fout.is_open()) {
    std::cerr << "Error opening file " << argv[3] << std::endl;
    exit(1);
  } else {
    std::cout << "Loading file " << argv[3] << std::endl;
  }
  for (int i = 0; i < q_cpu->nodesCount; ++i) fout << M[i] << '\n';

  cudaDeviceSynchronize();
  cudaDeviceReset();


  return 0;
}
