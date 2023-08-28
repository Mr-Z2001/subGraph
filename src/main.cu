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
  Graph *g_cpu, *q_cpu;
//  Graph *g_gpu, *q_gpu;
  g_cpu = new Graph();
  q_cpu = new Graph();

  // g_gpu
  int *g_partitionSize, *g_nodesCount, *g_edgesCount, *g_edgeLabelsCount, *g_idx2Size, *g_encodingSize;
  int *g_nodes, *g_row, *g_column, *g_edges_labels, *g_node_labels, *g_index1, *g_index2, *g_encoding;
  int *g_inDegree, *g_outDegree;

  // q_gpu
  int *q_partitionSize, *q_nodesCount, *q_edgesCount, *q_edgeLabelsCount, *q_idx2Size, *q_encodingSize, *q_candidateSize;
  int *q_nodes, *q_row, *q_column, *q_edges_labels, *q_node_labels, *q_index1, *q_index2, *q_encoding, *q_candidates, *q_order;
  double *q_score;
  int *q_inDegree, *q_outDegree;
  int *q_core, *q_forest;

#ifdef DEBUG
  puts("1");
#endif

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

  // g_gpu
  cudaStream_t streams[22];
  for (auto &i: streams) cudaStreamCreate(&i);
  cudaMallocAsync((void **) &g_partitionSize, sizeof(int), nullptr, streams[0]);
  cudaMallocAsync((void **) &g_nodesCount, sizeof(int), nullptr, streams[1]);
  cudaMallocAsync((void **) &g_edgesCount, sizeof(int), nullptr, streams[2]);
  cudaMallocAsync((void **) &g_edgeLabelsCount, sizeof(int), nullptr, streams[3]);
  cudaMallocAsync((void **) &g_idx2Size, sizeof(int), nullptr, streams[4]);
  cudaMallocAsync((void **) &g_encodingSize, sizeof(int), nullptr, streams[5]);
  cudaMallocAsync((void **) &g_nodes, g_cpu->nodesCount * sizeof(VID_t), nullptr, streams[6]);
  cudaMallocAsync((void **) &g_row, g_cpu->nodesCount * sizeof(int), nullptr, streams[7]);
  cudaMallocAsync((void **) &g_column, g_cpu->edgesCount * sizeof(int), nullptr, streams[8]);
  cudaMallocAsync((void **) &g_edges_labels, g_cpu->edgesCount * sizeof(Label_t), nullptr, streams[9]);
  cudaMallocAsync((void **) &g_node_labels, g_cpu->nodesCount * sizeof(Label_t), nullptr, streams[10]);
  cudaMallocAsync((void **) &g_index1, (g_cpu->nodesCount + 1) * sizeof(int), nullptr, streams[11]);
  cudaMallocAsync((void **) &g_index2, g_cpu->edgesCount * sizeof(int), nullptr, streams[12]);
  cudaMallocAsync((void **) &g_encoding, g_cpu->nodesCount * g_cpu->encodingSize * sizeof(int), nullptr,
                  streams[13]);
  cudaMallocAsync((void **) &g_inDegree, g_cpu->nodesCount * sizeof(int), nullptr, streams[14]);
  cudaMallocAsync((void **) &g_outDegree, g_cpu->nodesCount * sizeof(int), nullptr, streams[15]);

  for (auto &i: streams) cudaStreamSynchronize(i);
  // q_gpu
  cudaMallocAsync((void **) &q_partitionSize, sizeof(int), nullptr, streams[0]);
  cudaMallocAsync((void **) &q_nodesCount, sizeof(int), nullptr, streams[1]);
  cudaMallocAsync((void **) &q_edgesCount, sizeof(int), nullptr, streams[2]);
  cudaMallocAsync((void **) &q_edgeLabelsCount, sizeof(int), nullptr, streams[3]);
  cudaMallocAsync((void **) &q_idx2Size, sizeof(int), nullptr, streams[4]);
  cudaMallocAsync((void **) &q_encodingSize, sizeof(int), nullptr, streams[5]);
  cudaMallocAsync((void **) &q_candidateSize, sizeof(int), nullptr, streams[6]);
  cudaMallocAsync((void **) &q_nodes, q_cpu->nodesCount * sizeof(VID_t), nullptr, streams[7]);
  cudaMallocAsync((void **) &q_row, q_cpu->nodesCount * sizeof(int), nullptr, streams[8]);
  cudaMallocAsync((void **) &q_column, q_cpu->edgesCount * sizeof(int), nullptr, streams[9]);
  cudaMallocAsync((void **) &q_edges_labels, q_cpu->edgesCount * sizeof(Label_t), nullptr, streams[10]);
  cudaMallocAsync((void **) &q_node_labels, q_cpu->nodesCount * sizeof(Label_t), nullptr, streams[11]);
  cudaMallocAsync((void **) &q_index1, (q_cpu->nodesCount + 1) * sizeof(int), nullptr, streams[12]);
  cudaMallocAsync((void **) &q_index2, q_cpu->edgesCount * sizeof(int), nullptr, streams[13]);
  cudaMallocAsync((void **) &q_encoding, q_cpu->nodesCount * q_cpu->encodingSize * sizeof(int), nullptr,
                  streams[14]);
  cudaMallocAsync((void **) &q_candidates, q_cpu->nodesCount * g_cpu->nodesCount * sizeof(int), nullptr,
                  streams[15]);
  cudaMallocAsync((void **) &q_score, q_cpu->nodesCount * sizeof(double), nullptr, streams[16]);
  cudaMallocAsync((void **) &q_order, q_cpu->nodesCount * sizeof(int), nullptr, streams[17]);
  cudaMallocAsync((void **) &q_inDegree, q_cpu->nodesCount * sizeof(int), nullptr, streams[18]);
  cudaMallocAsync((void **) &q_outDegree, q_cpu->nodesCount * sizeof(int), nullptr, streams[19]);
  cudaMallocAsync((void **) &q_core, q_cpu->nodesCount * sizeof(int), nullptr, streams[20]);
  cudaMallocAsync((void **) &q_forest, q_cpu->nodesCount * sizeof(int), nullptr, streams[21]);
  for (auto &i: streams) cudaStreamSynchronize(i);


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
  // q
  cudaMemcpyAsync(q_partitionSize, &q_cpu->partitionSize, sizeof(int), cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(q_nodesCount, &q_cpu->nodesCount, sizeof(int), cudaMemcpyHostToDevice, streams[1]);
  cudaMemcpyAsync(q_edgesCount, &q_cpu->edgesCount, sizeof(int), cudaMemcpyHostToDevice, streams[2]);
  cudaMemcpyAsync(q_edgeLabelsCount, &q_cpu->edgeLabelsCount, sizeof(int), cudaMemcpyHostToDevice, streams[3]);
//  cudaMemcpyAsync(q_idx2Size, &q_cpu->idx2Size, sizeof(int), cudaMemcpyHostToDevice, streams[4]);
//  cudaMemcpyAsync(q_encodingSize, &q_cpu->encodingSize, sizeof(int), cudaMemcpyHostToDevice, streams[5]);
  cudaMemcpyAsync(q_nodes, q_cpu->nodes, q_cpu->nodesCount * sizeof(VID_t), cudaMemcpyHostToDevice, streams[6]);
  cudaMemcpyAsync(q_row, q_cpu->row, q_cpu->nodesCount * sizeof(int), cudaMemcpyHostToDevice, streams[7]);
  cudaMemcpyAsync(q_column, q_cpu->column, q_cpu->edgesCount * sizeof(int), cudaMemcpyHostToDevice, streams[8]);
  cudaMemcpyAsync(q_edges_labels, q_cpu->edges_labels, q_cpu->edgesCount * sizeof(Label_t), cudaMemcpyHostToDevice,
                  streams[9]);
  cudaMemcpyAsync(q_node_labels, q_cpu->node_labels, q_cpu->nodesCount * sizeof(Label_t), cudaMemcpyHostToDevice,
                  streams[10]);
  cudaMemcpyAsync(q_inDegree, q_cpu->inDegree, q_cpu->nodesCount * sizeof(int), cudaMemcpyHostToDevice, streams[11]);
  cudaMemcpyAsync(q_outDegree, q_cpu->outDegree, q_cpu->nodesCount * sizeof(int), cudaMemcpyHostToDevice,
                  streams[12]);
  for (int i = 0; i < 13; ++i) cudaStreamSynchronize(streams[i]);

  // g
  cudaMemcpyAsync(g_partitionSize, &g_cpu->partitionSize, sizeof(int), cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(g_nodesCount, &g_cpu->nodesCount, sizeof(int), cudaMemcpyHostToDevice, streams[1]);
  cudaMemcpyAsync(g_edgesCount, &g_cpu->edgesCount, sizeof(int), cudaMemcpyHostToDevice, streams[2]);
  cudaMemcpyAsync(g_edgeLabelsCount, &g_cpu->edgeLabelsCount, sizeof(int), cudaMemcpyHostToDevice, streams[3]);
//  cudaMemcpyAsync(g_idx2Size, &g_cpu->idx2Size, sizeof(int), cudaMemcpyHostToDevice, streams[4]);
  // cudaMemcpyAsync(g_encodingSize, &g_cpu->encodingSize, sizeof(int), cudaMemcpyHostToDevice, streams[5]);
  cudaMemcpyAsync(g_nodes, g_cpu->nodes, g_cpu->nodesCount * sizeof(VID_t), cudaMemcpyHostToDevice, streams[6]);
  cudaMemcpyAsync(g_row, g_cpu->row, g_cpu->nodesCount * sizeof(int), cudaMemcpyHostToDevice, streams[7]);
  cudaMemcpyAsync(g_column, g_cpu->column, g_cpu->edgesCount * sizeof(int), cudaMemcpyHostToDevice, streams[8]);
  cudaMemcpyAsync(g_edges_labels, g_cpu->edges_labels, g_cpu->edgesCount * sizeof(Label_t), cudaMemcpyHostToDevice,
                  streams[9]);
  cudaMemcpyAsync(g_node_labels, g_cpu->node_labels, g_cpu->nodesCount * sizeof(Label_t), cudaMemcpyHostToDevice,
                  streams[10]);
  cudaMemcpyAsync(g_inDegree, g_cpu->inDegree, g_cpu->nodesCount * sizeof(int), cudaMemcpyHostToDevice, streams[11]);
  cudaMemcpyAsync(g_outDegree, g_cpu->outDegree, g_cpu->nodesCount * sizeof(int), cudaMemcpyHostToDevice,
                  streams[12]);
  for (int i = 0; i < 13; ++i) cudaStreamSynchronize(streams[i]);


#ifdef INFO
  std::cout << "graph copied to GPU" << std::endl;
#endif

  ///////////////Filter//////////////////////
//  int *g_labels_d, *q_labels_d;
//  cudaMalloc((void **) &g_labels_d, g_cpu->nodesCount * sizeof(int));
//  cudaMalloc((void **) &q_labels_d, q_cpu->nodesCount * sizeof(int));

  getEncoding<<<grid, block, 0, streams[0]>>>(g_nodesCount, g_edgesCount, g_edges_labels, g_index1,
                                              g_index2, g_encoding);
  getEncoding<<<grid, block, 0, streams[1]>>>(q_nodesCount, q_edgesCount, q_edges_labels, q_index1,
                                              q_index2, q_encoding);
  cudaDeviceSynchronize();

#ifdef DEBUG
  cudaMemcpyAsync(g_cpu->encoding, g_encoding, sizeof(Graph), cudaMemcpyDeviceToHost, streams[0]);
  cudaMemcpyAsync(q_cpu->encoding, q_encoding, sizeof(Graph), cudaMemcpyDeviceToHost, streams[1]);
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

  firstFiltering<<<grid, block>>>(g_nodesCount, g_edgesCount, g_encoding,
                                  q_nodesCount, q_edgesCount, q_encoding, q_candidates);
  cudaDeviceSynchronize();
  cudaMemcpy(q_cpu->candidates, q_candidates, sizeof(Graph), cudaMemcpyDeviceToHost);


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

  cudaMemcpyAsync(q_forest, q_cpu->forest, sizeof(Graph), cudaMemcpyHostToDevice, streams[0]);
  cudaMemcpyAsync(q_core, q_cpu->core, sizeof(Graph), cudaMemcpyHostToDevice, streams[1]);
  for (int i = 0; i < 2; ++i) cudaStreamSynchronize(streams[i]);
//  for (auto &i: streams) cudaStreamSynchronize(i);

#ifdef INFO
  std::cout << "core-forest decomposition done." << std::endl;
#endif
  // get score - determine the order of join
  getScore<<<grid, block>>>(q_nodesCount, q_candidateSize, q_candidates, q_inDegree, q_outDegree, q_score);
  cudaDeviceSynchronize();
  std::cout << "OK" << std::endl;

#ifdef DEBUG
  cudaMemcpy(q_cpu->score, q_score, sizeof(Graph), cudaMemcpyDeviceToHost);
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

//  cudaMemcpyAsync(q_cpu, q_gpu, sizeof(Graph), cudaMemcpyDeviceToHost, streams[0]);
//  cudaMemcpyAsync(g_cpu, g_gpu, sizeof(Graph), cudaMemcpyDeviceToHost, streams[1]);
//  cudaStreamSynchronize(streams[0]);
//  cudaStreamSynchronize(streams[1]);

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
