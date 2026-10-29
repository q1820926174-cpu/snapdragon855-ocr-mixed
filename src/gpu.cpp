// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 q1820926174-cpu
// Single managed accelerator owner; bounded closed-loop requests and independent GPU consumers.
#include <net.h>
#include <gpu.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <deque>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
double ms(double t){return (now()-t)*1000;}
std::string env(const char*k,const char*def=""){auto*v=std::getenv(k);return v?v:def;}
int integer(const char*k,int def){auto s=env(k);return s.empty()?def:std::stoi(s);}
void check(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);}
std::string quote(const std::string&s){std::string r="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=c;}else if(c<32)r+=' ';else r+=c;}return r+'"';}
struct Row{std::string id,path;};
struct Job{Row row;double start=0,front=0,bridge=0,ready=0,front_start=0,front_end=0; ncnn::Mat x;};
struct Decoded{std::string text;uint64_t hash=1469598103934665603ULL;double confidence=0;int chars=0;};
Decoded decode_output(const ncnn::Mat&y,const std::vector<std::string>&keys){
 Decoded d;size_t previous=0;const float*data=y;
 for(int t=0;t<80;++t){size_t best=0;float max=-INFINITY;for(int k=0;k<18710;++k){float v=data[t*18710+k];if(!std::isfinite(v))throw std::runtime_error("nonfinite output");if(v>max){max=v;best=k;}}uint32_t bits;std::memcpy(&bits,&max,4);d.hash^=uint32_t(best);d.hash*=1099511628211ULL;d.hash^=bits;d.hash*=1099511628211ULL;if(best&&best!=previous){d.text+=keys[best];d.confidence+=max;++d.chars;}previous=best;}
 if(d.chars)d.confidence/=d.chars;return d;
}
#include <command.h>
#include <pipeline.h>
// One 64-lane workgroup per time step; preserve first-index tie semantics.
static const char* scv41_ctc_shader=R"glsl(#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input {float data[];} input_data;
layout(binding=1) writeonly buffer Output {float data[];} output_data;
layout(push_constant) uniform parameter {int classes;int rows;int logits;int pack;} p;
shared float values[64];shared uint indices[64];shared uint invalid[64];
void main(){uint lane=gl_LocalInvocationID.x;uint row=gl_WorkGroupID.x;if(row>=uint(p.rows))return;
float maximum=-3.402823466e38;uint best=0xffffffffu;uint bad=0u;
uint base=(row/uint(p.pack))*uint(p.classes)*uint(p.pack)+(row%uint(p.pack));
for(uint j=lane;j<uint(p.classes);j+=64u){float v=input_data.data[base+j*uint(p.pack)];if((floatBitsToUint(v)&0x7f800000u)==0x7f800000u)bad=1u;if(v>maximum||(v==maximum&&j<best)){maximum=v;best=j;}}
values[lane]=maximum;indices[lane]=best;invalid[lane]=bad;barrier();
for(uint offset=32u;offset>0u;offset>>=1u){if(lane<offset){float other=values[lane+offset];uint index=indices[lane+offset];if(other>values[lane]||(other==values[lane]&&index<indices[lane])){values[lane]=other;indices[lane]=index;}invalid[lane]|=invalid[lane+offset];}barrier();}
float selected=values[0];uint selected_index=indices[0];uint any_bad=invalid[0];barrier();
if(p.logits!=0){float sum=0.0;for(uint j=lane;j<uint(p.classes);j+=64u)sum+=exp(input_data.data[base+j*uint(p.pack)]-selected);values[lane]=sum;barrier();for(uint offset=32u;offset>0u;offset>>=1u){if(lane<offset)values[lane]+=values[lane+offset];barrier();}selected=1.0/values[0];}
if(lane==0u){output_data.data[row*2u]=any_bad!=0u?-1.0:float(selected_index);output_data.data[row*2u+1u]=selected;}}
)glsl";
struct FusedGpuCtc{
 const ncnn::VulkanDevice*device; ncnn::VkAllocator*blob; ncnn::VkAllocator*staging;ncnn::Option opt;std::unique_ptr<ncnn::Pipeline>pipeline;
 FusedGpuCtc(const ncnn::Option&options):device(ncnn::get_gpu_device(0)),blob(device->acquire_blob_allocator()),staging(device->acquire_staging_allocator()),opt(options){
  opt.use_fp16_packed=false;opt.use_fp16_storage=false;opt.use_fp16_arithmetic=false;opt.use_fp16_uniform=false;opt.use_packing_layout=false;opt.blob_vkallocator=blob;opt.workspace_vkallocator=blob;opt.staging_vkallocator=staging;
  std::vector<uint32_t>spv;check(!ncnn::compile_spirv_module(scv41_ctc_shader,opt,spv),"fused CTC shader compile");pipeline.reset(new ncnn::Pipeline(device));pipeline->set_subgroup_size(64);pipeline->set_local_size_xyz(64,1,1);check(pipeline->local_size_x()==64&&pipeline->local_size_y()==1&&pipeline->local_size_z()==1,"fused CTC workgroup mismatch");check(!pipeline->create(spv.data(),spv.size()*4,{}),"fused CTC pipeline create");
 }
 ~FusedGpuCtc(){pipeline.reset();device->reclaim_blob_allocator(blob);device->reclaim_staging_allocator(staging);}
 Decoded execute(ncnn::Extractor&extractor,const std::vector<std::string>&keys,bool logits,ncnn::Extractor*prefix=nullptr){
  extractor.set_blob_vkallocator(blob);extractor.set_workspace_vkallocator(blob);extractor.set_staging_vkallocator(staging);ncnn::VkCompute command(device);
#if NCNN_BENCHMARK
  command.create_query_pool(4096);
#endif
  if(prefix){
   prefix->set_blob_vkallocator(blob);prefix->set_workspace_vkallocator(blob);prefix->set_staging_vkallocator(staging);
   ncnn::VkMat half_feature,feature;check(!prefix->extract("t_293",half_feature,command),"resident CNN extraction");
   device->convert_packing(half_feature,feature,4,1,command,opt);check(feature.elembits()==32,"resident FP32 feature");
   check(!extractor.input("t_293",feature),"resident tail input");
  }
  ncnn::VkMat input;
  check(!extractor.extract(logits?"t_469":"t_470",input,command),"fused CTC GPU extraction");
  check(input.w==18710&&input.h*input.elempack==80&&input.c==1&&input.n==1&&input.elembits()==32,"fused CTC GPU input layout");
  if(std::getenv("SCV41_GPU_CTC_DEBUG")){ncnn::Mat wide;command.record_clone(input,wide,opt);check(!command.submit_and_wait(),"debug full input download");ncnn::Mat unpacked;ncnn::Option o;o.num_threads=1;ncnn::convert_packing(wide,unpacked,1,o);const float*p=unpacked;size_t bad=0;for(size_t i=0;i<size_t(80)*18710;++i)bad+=!std::isfinite(p[i]);std::cerr<<"FUSED_INPUT_DEBUG pack="<<input.elempack<<" dims="<<input.dims<<" finite_bad="<<bad<<std::endl;check(!bad,"full GPU input is not finite");command.reset();}
  ncnn::VkMat small(2,80,size_t(4),blob);ncnn::VkMat dispatch;dispatch.w=80*64;dispatch.h=1;dispatch.c=1;std::vector<ncnn::vk_constant_type>constants(4);constants[0].i=18710;constants[1].i=80;constants[2].i=logits?1:0;constants[3].i=input.elempack;
  command.record_pipeline(pipeline.get(),{input,small},constants,dispatch);ncnn::Mat host;command.record_download(small,host,opt);check(!command.submit_and_wait(),"fused CTC submit");if(host.elempack!=1){ncnn::Mat unpacked;ncnn::Option cpu_opt;cpu_opt.num_threads=1;ncnn::convert_packing(host,unpacked,1,cpu_opt);host=unpacked;}check(host.w==2&&host.h==80&&host.c==1&&host.elembits()==32,"fused CTC host shape "+std::to_string(host.w)+"x"+std::to_string(host.h)+" total="+std::to_string(host.total()));
  const float*data=host;Decoded d;int previous=0;
  for(int t=0;t<80;++t){const float fi=data[t*2],prob=data[t*2+1];int index=int(std::lround(fi));check(std::isfinite(fi)&&std::isfinite(prob)&&index>=0&&index<18710&&std::abs(fi-index)<.001&&prob>=0&&prob<=1.001,"fused CTC invalid result row="+std::to_string(t)+" index="+std::to_string(fi)+" prob="+std::to_string(prob));
   uint32_t bits;std::memcpy(&bits,&prob,4);d.hash^=uint32_t(index);d.hash*=1099511628211ULL;d.hash^=bits;d.hash*=1099511628211ULL;
   if(index&&index!=previous){d.text+=keys[index];d.confidence+=prob;++d.chars;}previous=index;}
  if(d.chars)d.confidence/=d.chars;return d;
 }
};


Decoded decode_logits(const ncnn::Mat&input,const std::vector<std::string>&keys){
 ncnn::Mat y=input;if(y.elempack!=1){ncnn::Mat unpacked;ncnn::Option o;ncnn::convert_packing(y,unpacked,1,o);y=unpacked;}
 check(y.w==18710&&y.h==80&&y.c==1&&y.elembits()==32,"CPU logits shape");Decoded d;int previous=0;const float*p=y;
 for(int t=0;t<80;++t){int best=0;float maximum=-INFINITY;for(int c=0;c<18710;++c){float v=p[t*18710+c];check(std::isfinite(v),"nonfinite CPU logits");if(v>maximum){maximum=v;best=c;}}double sum=0;for(int c=0;c<18710;++c)sum+=std::exp(double(p[t*18710+c])-maximum);float probability=float(1/sum);uint32_t bits;std::memcpy(&bits,&probability,4);d.hash^=uint32_t(best);d.hash*=1099511628211ULL;d.hash^=bits;d.hash*=1099511628211ULL;if(best&&best!=previous){d.text+=keys[best];d.confidence+=probability;++d.chars;}previous=best;}
 if(d.chars)d.confidence/=d.chars;return d;
}
int main(){try{
 const std::string base=env("SCV41_PIPE_BASE");const bool cpu=env("SCV41_PIPE_ENGINE")=="cpu",split=!cpu&&integer("SCV41_PNNX_SPLIT",1)!=0;const int workers=integer("SCV41_PIPE_WORKERS",2),capacity=integer("SCV41_PIPE_CONCURRENCY",10),limit=integer("SCV41_PIPE_LIMIT",40),offset=integer("SCV41_PIPE_OFFSET",0);check(workers>=1&&workers<=10&&capacity>=1&&capacity<=10&&limit>=1,"scheduling options");
 std::vector<std::string>keys;std::string line;std::ifstream dict(base+"/keys.txt");while(std::getline(dict,line))keys.push_back(line);check(keys.size()==18710,"dictionary");std::vector<Row>rows;int index=0;std::ifstream list(env("SCV41_PIPE_INPUT_LIST",(base+"/all.tsv").c_str()));while(std::getline(list,line)&&int(rows.size())<limit){if(index++<offset)continue;auto tab=line.find('\t');check(tab!=std::string::npos,"input list");rows.push_back({line.substr(0,tab),line.substr(tab+1)});}check(!rows.empty(),"empty input list");
 auto initialize=now();if(!cpu)check(ncnn::get_gpu_count()==1,"GPU count");std::vector<std::unique_ptr<ncnn::Net>>cnns,tails;std::vector<std::unique_ptr<FusedGpuCtc>>reducers;
 for(int i=0;i<workers;++i){
  auto tail=std::unique_ptr<ncnn::Net>(new ncnn::Net);tail->opt.num_threads=integer("SCV41_PIPE_CPU_THREADS",1);tail->opt.use_vulkan_compute=!cpu;tail->opt.use_fp16_packed=false;tail->opt.use_fp16_storage=false;tail->opt.use_fp16_arithmetic=false;tail->opt.use_fp16_uniform=false;tail->opt.use_subgroup_ops=false;tail->opt.use_cooperative_matrix=false;if(!cpu)tail->set_vulkan_device(0);
  auto cnn=std::unique_ptr<ncnn::Net>();if(split){cnn.reset(new ncnn::Net);cnn->opt=tail->opt;cnn->opt.use_fp16_packed=integer("SCV41_PNNX_CNN_FP32",0)==0;cnn->opt.use_fp16_arithmetic=integer("SCV41_PNNX_CNN_HALF_ARITH",0)!=0;cnn->set_vulkan_device(0);check(!cnn->load_param((base+"/cnn/model.param").c_str())&&!cnn->load_model((base+"/cnn/model.bin").c_str()),"CNN model");}
  std::string part=split?"/tail":"/full";check(!tail->load_param((base+part+"/model.param").c_str())&&!tail->load_model((base+part+"/model.bin").c_str()),"tail/full model");reducers.emplace_back(cpu?nullptr:new FusedGpuCtc(tail->opt));cnns.push_back(std::move(cnn));tails.push_back(std::move(tail));
 }
 std::cout<<"PIPE_INITIALIZED engine="<<(cpu?"cpu":"gpu")<<" npu_contexts=0 workers="<<workers<<" batch=1 concurrency="<<capacity<<" init_ms="<<ms(initialize)<<" pnnx=1 resident_bridge="<<split<<" feature_host_readbacks=0"<<std::endl;
 std::ofstream output(env("SCV41_PIPE_OUTPUT"));check(bool(output),"output file");std::mutex mutex,write_mutex;std::condition_variable cv;std::deque<Job>queue;bool done=false;std::atomic<bool>failed{false};std::string error;size_t completed=0,admitted=0,peak=0;auto fail=[&](const std::string&why){std::lock_guard<std::mutex>lock(mutex);if(!failed.exchange(true))error=why;cv.notify_all();};
 std::this_thread::sleep_for(std::chrono::seconds(6));double run=now();std::vector<std::thread>threads;
 for(int worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{try{while(true){Job job;{std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return failed||done||!queue.empty();});if(failed||(queue.empty()&&done))return;job=std::move(queue.front());queue.pop_front();}
   double start=now();Decoded decoded;auto tail=tails[worker]->create_extractor();if(cpu){check(!tail.input("t_0",job.x),"CPU input");ncnn::Mat y;check(!tail.extract("t_469",y),"CPU extraction");decoded=decode_logits(y,keys);}else if(split){auto prefix=cnns[worker]->create_extractor();check(!prefix.input("t_0",job.x),"CNN input");decoded=reducers[worker]->execute(tail,keys,true,&prefix);}else{check(!tail.input("t_0",job.x),"GPU input");decoded=reducers[worker]->execute(tail,keys,true);}
   double end=now(),gpu_ms=(end-start)*1000;{std::lock_guard<std::mutex>lock(write_mutex);output<<std::setprecision(14)<<"{\"id\":"<<quote(job.row.id)<<",\"text\":"<<quote(decoded.text)<<",\"hash\":\""<<std::hex<<decoded.hash<<std::dec<<"\",\"confidence\":"<<decoded.confidence<<",\"fallback\":false,\"fallback_ms\":0,\"worker\":"<<worker<<",\"start_s\":"<<job.start<<",\"end_s\":"<<end<<",\"front_start_s\":0,\"front_end_s\":0,\"gpu_start_s\":"<<start<<",\"gpu_end_s\":"<<end<<",\"front_ms\":0,\"bridge_ms\":0,\"gpu_ms\":"<<gpu_ms<<",\"queue_ms\":"<<(start-job.ready)*1000<<",\"decode_ms\":0,\"total_ms\":"<<(end-job.start)*1000<<"}\n";output.flush();check(bool(output),"output write");}{std::lock_guard<std::mutex>lock(mutex);++completed;cv.notify_all();}
 }}catch(const std::exception&e){fail(e.what());}});
 try{for(const auto&row:rows){Job job;job.row=row;{std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return failed||admitted-completed<size_t(capacity);});check(!failed,"consumer failed");job.start=now();++admitted;peak=std::max(peak,admitted-completed);}std::vector<float>raw(92160);std::ifstream input(row.path,std::ios::binary);input.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);check(bool(input),"input read");job.x=ncnn::Mat(640,48,3);for(int c=0;c<3;++c){float*p=job.x.channel(c);for(int s=0;s<48*640;++s)p[s]=raw[s*3+c];}job.ready=now();{std::lock_guard<std::mutex>lock(mutex);queue.push_back(std::move(job));cv.notify_all();}}}catch(const std::exception&e){fail(e.what());}
 {std::lock_guard<std::mutex>lock(mutex);done=true;cv.notify_all();}for(auto&t:threads)t.join();check(!failed,"PIPE_FAILED "+error);check(completed==rows.size(),"incomplete output");std::cout<<"PIPE_COMPLETE "<<completed<<" duration_ms="<<ms(run)<<" peak_inflight="<<peak<<" feature_host_readbacks=0 ctc_readback_bytes=640"<<std::endl;std::this_thread::sleep_for(std::chrono::seconds(6));return 0;
}catch(const std::exception&e){std::cerr<<"PIPE_ERROR "<<e.what()<<std::endl;return 70;}}
