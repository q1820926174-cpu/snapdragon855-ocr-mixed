// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 q1820926174-cpu
// Single managed accelerator owner; bounded closed-loop requests and independent GPU consumers.
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/TensorShapeMap.hpp>
#include <DlSystem/IBufferAttributes.hpp>
#include <DlSystem/IUserBuffer.hpp>
#include <DlSystem/TensorMap.hpp>
#include <DiagLog/IDiagLog.hpp>
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
 Decoded execute(ncnn::Extractor&extractor,const std::vector<std::string>&keys,bool logits){
  extractor.set_blob_vkallocator(blob);extractor.set_workspace_vkallocator(blob);extractor.set_staging_vkallocator(staging);ncnn::VkCompute command(device);
#if NCNN_BENCHMARK
  command.create_query_pool(4096);
#endif
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

#include <layer.h>
#include <modelbin.h>
#include <command.h>
#include <pipeline.h>
#include <future>
#include <functional>
#include <sstream>
// All SNPE models are built/executed/destroyed by one owner, sharing the process RPC session.
class Sm8150Owner {
 std::mutex mutex;std::condition_variable cv;std::deque<std::function<void()>>tasks;bool stop=false;std::thread worker;
public:
 Sm8150Owner():worker([this]{while(true){std::function<void()>f;{std::unique_lock<std::mutex>l(mutex);cv.wait(l,[this]{return stop||!tasks.empty();});if(stop&&tasks.empty())return;f=std::move(tasks.front());tasks.pop_front();}f();}}){}
 ~Sm8150Owner(){{std::lock_guard<std::mutex>l(mutex);stop=true;cv.notify_all();}worker.join();}
 template<class F>auto call(F f)->decltype(f()) {using R=decltype(f());auto task=std::make_shared<std::packaged_task<R()>>(std::move(f));auto future=task->get_future();{std::lock_guard<std::mutex>l(mutex);tasks.push_back([task]{(*task)();});cv.notify_all();}return future.get();}
};
static Sm8150Owner&sm8150_owner(){static Sm8150Owner owner;return owner;}
static std::atomic<size_t>sm8150_executions{0};
static std::atomic<uint64_t>sm8150_execute_us{0};
static bool sm8150_enabled(int id){auto ids=env("SCV41_855_NPU_IDS","all");if(ids=="all")return true;return (","+ids+",").find(","+std::to_string(id)+",")!=std::string::npos;}
static void sm8150_rpc(){static void*handle=nullptr;if(handle)return;auto path=env("SCV41_NPU_RPC_LIBRARY");check(!path.empty(),"SM8150 managed NPU wrapper required");handle=dlopen(path.c_str(),RTLD_NOW|RTLD_GLOBAL);check(handle,"SM8150 RPC library");auto*p=static_cast<char*>(dlsym(handle,"DSP_LIBS_LOCATION"));check(p,"SM8150 RPC search path");const char*search="/opt/scv41-snpe-1.59.0.3230/dsp-original/cdsp;/opt/scv41-snpe-1.59.0.3230/lib/dsp;";std::memcpy(p,search,std::strlen(search)+1);}
#include <DlSystem/IUserBufferFactory.hpp>
#include <DlSystem/UserBufferMap.hpp>
#include <map>
#include <sys/stat.h>
static std::atomic<int>sm8150_context_count{0};
struct Sm8150Tf8Model {
 std::unique_ptr<zdl::DlContainer::IDlContainer>container;std::unique_ptr<zdl::SNPE::SNPE>net;
 std::vector<uint8_t>input,output;std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>buffers;zdl::DlSystem::UserBufferMap in,out;
 float input_step=0,output_step=0;int input_zero=0,output_zero=0;int ci,co,w,h;
 Sm8150Tf8Model(const std::string&folder,int ic,int oc,int iw,int ih):ci(ic),co(oc),w(iw),h(ih){
  sm8150_rpc();container=zdl::DlContainer::IDlContainer::open((folder+"/int8-hta.dlc").c_str());check(bool(container),"TF8 DLC "+folder);
  net=zdl::SNPE::SNPEBuilder(container.get()).setRuntimeProcessor(zdl::DlSystem::Runtime_t::AIP_FIXED8_TF).setCPUFallbackMode(false).setUseUserSuppliedBuffers(true).setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE).build();check(bool(net),std::string("TF8 model build ")+zdl::DlSystem::getLastErrorString());
  for(int direction=0;direction<2;++direction){const char*name=direction?"y":"x";auto attrs=net->getInputOutputBufferAttributes(name);check(bool(attrs),"TF8 attributes");auto dims=(*attrs)->getDims();check(dims.rank()==4&&dims[0]==1&&dims[1]==size_t(h)&&dims[2]==size_t(w)&&dims[3]==size_t(direction?co:ci),"TF8 requires verified NHWC");auto*enc=dynamic_cast<zdl::DlSystem::UserBufferEncodingTfN*>((*attrs)->getEncoding());check(enc&&enc->getElementSize()==1,"TF8 encoding type");float step=enc->getQuantizedStepSize();int zero=enc->getStepExactly0();check(std::isfinite(step)&&step>0&&zero>=0&&zero<=255,"TF8 encoding range");auto&data=direction?output:input;data.resize(size_t(direction?co:ci)*w*h);std::vector<size_t>strides(dims.rank(),1);size_t count=1;for(size_t j=dims.rank();j>0;--j){strides[j-1]=count;count*=dims[j-1];}zdl::DlSystem::UserBufferEncodingTfN encoding(uint64_t(zero),step,8);auto buffer=zdl::SNPE::SNPEFactory::getUserBufferFactory().createUserBuffer(data.data(),data.size(),strides,&encoding);check(bool(buffer),"TF8 buffer creation");if(direction){output_step=step;output_zero=zero;out.add(name,buffer.get());}else{input_step=step;input_zero=zero;in.add(name,buffer.get());}buffers.push_back(std::move(buffer));}
  ++sm8150_context_count;std::cerr<<"SM8150_TF8_LOADED "<<folder<<" input_step="<<input_step<<" input_zero="<<input_zero<<" output_step="<<output_step<<" output_zero="<<output_zero<<" cpu_fallback=disabled"<<std::endl;
 }
 ~Sm8150Tf8Model(){--sm8150_context_count;}
 std::vector<uint8_t>execute(const ncnn::Mat&words){check(words.elempack==1&&words.w*4==int(input.size()),"TF8 input byte count");std::memcpy(input.data(),words.data,input.size());auto t=now();check(net->execute(in,out),std::string("TF8 execute ")+zdl::DlSystem::getLastErrorString());sm8150_execute_us.fetch_add(uint64_t(ms(t)*1000));++sm8150_executions;return output;}
};
static std::shared_ptr<Sm8150Tf8Model>sm8150_get_model(const std::string&folder,int ci,int co,int w,int h){static std::map<std::string,std::weak_ptr<Sm8150Tf8Model>>cache;auto model=cache[folder].lock();if(!model){model=std::make_shared<Sm8150Tf8Model>(folder,ci,co,w,h);cache[folder]=model;}return model;}
static const char*sm8150_quant_shader=R"glsl(#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input {float v[];} src;
layout(binding=1) readonly buffer Coeff {float v[];} coeff;
layout(binding=2) writeonly buffer Output {uint v[];} dst;
layout(push_constant) uniform parameter {int spatial;int channels;int cstep;float step;int zero;} p;
void main(){uint i=gl_GlobalInvocationID.x;uint words=uint(p.spatial*p.channels)/4u;if(i>=words)return;uint s=i/(uint(p.channels)/4u),c=(i%(uint(p.channels)/4u))*4u;uint packed=0u;for(uint k=0u;k<4u;++k){float v=(src.v[(c+k)*uint(p.cstep)+s]-coeff.v[uint(p.channels)+c+k])*coeff.v[c+k];uint q=uint(clamp(roundEven(v/p.step)+float(p.zero),0.0,255.0));packed|=q<<(8u*k);}dst.v[i]=packed;}
)glsl";
static const char*sm8150_dequant_shader=R"glsl(#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input {uint v[];} src;
layout(binding=1) readonly buffer Coeff {float v[];} coeff;
layout(binding=2) writeonly buffer Output {float v[];} dst;
layout(binding=3) readonly buffer OriginalInput {float v[];} original_input;
layout(binding=4) readonly buffer SparseCorrection {uint v[];} sparse;
layout(push_constant) uniform parameter {int spatial;int channels;int cstep;float step;int zero;int input_cstep;int nnz;} p;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(p.spatial*p.channels))return;uint c=i/uint(p.spatial),s=i%uint(p.spatial);uint index=s*uint(p.channels)+c;uint q=(src.v[index/4u]>>((index%4u)*8u))&255u;float v=(float(q)-float(p.zero))*p.step;float correction=0.0;uint begin=sparse.v[c],end=sparse.v[c+1u];for(uint e=begin;e<end;++e){uint column=sparse.v[uint(p.channels)+1u+e];float weight=uintBitsToFloat(sparse.v[uint(p.channels)+1u+uint(p.nnz)+e]);float mean=uintBitsToFloat(sparse.v[uint(p.channels)+1u+uint(p.nnz)*2u+column]);correction+=(original_input.v[column*uint(p.input_cstep)+s]-mean)*weight;}dst.v[c*uint(p.cstep)+s]=v*coeff.v[c]+coeff.v[uint(p.channels)+c]+correction;}
)glsl";
static const char*sm8150_quant_packed=R"glsl(#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input {uint v[];} src;
layout(binding=1) readonly buffer Coeff {float v[];} coeff;
layout(binding=2) writeonly buffer Output {uint v[];} dst;
layout(push_constant) uniform parameter {int spatial;int channels;int cstep;float step;int zero;} p;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(p.spatial*p.channels)/4u)return;uint s=i/(uint(p.channels)/4u),c=(i%(uint(p.channels)/4u))*4u;uint index=(c/4u)*uint(p.cstep)*2u+s*2u;vec2 lo=unpackHalf2x16(src.v[index]),hi=unpackHalf2x16(src.v[index+1u]);vec4 x=vec4(lo,hi);uint packed=0u;for(uint k=0u;k<4u;++k){float value=(x[k]-coeff.v[uint(p.channels)+c+k])*coeff.v[c+k];uint q=uint(clamp(roundEven(value/p.step)+float(p.zero),0.0,255.0));packed|=q<<(8u*k);}dst.v[i]=packed;}
)glsl";
static const char*sm8150_dequant_packed=R"glsl(#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
layout(binding=0) readonly buffer Input {uint v[];} src;
layout(binding=1) readonly buffer Coeff {float v[];} coeff;
layout(binding=2) writeonly buffer Output {uint v[];} dst;
layout(binding=3) readonly buffer OriginalInput {uint v[];} original_input;
layout(binding=4) readonly buffer SparseCorrection {uint v[];} sparse;
layout(push_constant) uniform parameter {int spatial;int channels;int cstep;float step;int zero;int input_cstep;int nnz;} p;
float input_value(uint channel,uint s){uint index=(channel/4u)*uint(p.input_cstep)*2u+s*2u+(channel%4u)/2u;vec2 values=unpackHalf2x16(original_input.v[index]);return values[channel%2u];}
float restore(uint c,uint s){uint index=s*uint(p.channels)+c;uint q=(src.v[index/4u]>>((index%4u)*8u))&255u;float value=(float(q)-float(p.zero))*p.step;float correction=0.0;uint begin=sparse.v[c],end=sparse.v[c+1u];for(uint e=begin;e<end;++e){uint column=sparse.v[uint(p.channels)+1u+e];float weight=uintBitsToFloat(sparse.v[uint(p.channels)+1u+uint(p.nnz)+e]);float mean=uintBitsToFloat(sparse.v[uint(p.channels)+1u+uint(p.nnz)*2u+column]);correction+=(input_value(column,s)-mean)*weight;}return value*coeff.v[c]+coeff.v[uint(p.channels)+c]+correction;}
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(p.spatial*p.channels)/2u)return;uint c=(i/uint(p.spatial))*2u,s=i%uint(p.spatial);uint index=(c/4u)*uint(p.cstep)*2u+s*2u+(c%4u)/2u;dst.v[index]=packHalf2x16(vec2(restore(c,s),restore(c+1u,s)));}
)glsl";
class NpuConv855:public ncnn::Layer {
 int id=0,ci=0,co=0,w=0,h=0,weight_count=0;bool active=false;std::unique_ptr<ncnn::Layer>gpu,teacher;ncnn::ParamDict original_params;mutable std::vector<double>stats;ncnn::Mat in_coeff,out_coeff,csr;ncnn::VkMat in_gpu,out_gpu,csr_gpu;int csr_nnz=0;std::unique_ptr<ncnn::Pipeline>quant,dequant,packed_quant,packed_dequant;std::shared_ptr<Sm8150Tf8Model>model;
 static ncnn::Option fp32(const ncnn::Option&opt){auto f=opt;f.use_fp16_packed=false;f.use_fp16_storage=false;f.use_fp16_arithmetic=false;f.use_fp16_uniform=false;f.use_bf16_storage=false;f.use_bf16_packed=false;f.use_packing_layout=false;return f;}
public:
 NpuConv855(){one_blob_only=true;support_inplace=false;support_vulkan=true;support_vulkan_packing=true;support_vulkan_any_packing=false;}
 ~NpuConv855(){if(!stats.empty()){auto dir=env("SCV41_855_CALIBRATE");mkdir(dir.c_str(),0700);std::ofstream f(dir+"/n_"+std::to_string(id)+".f64",std::ios::binary);f.write(reinterpret_cast<const char*>(stats.data()),stats.size()*sizeof(double));std::cerr<<"SM8150_CALIBRATION_STATS "<<id<<" samples_per_channel="<<stats[5]<<" saved="<<bool(f)<<std::endl;}if(model)sm8150_owner().call([this]{model.reset();});}
 int load_param(const ncnn::ParamDict&pd)override {original_params=pd;id=pd.get(26,-1);ci=pd.get(27,0);co=pd.get(0,0);w=pd.get(28,0);h=pd.get(29,0);weight_count=pd.get(6,0);active=sm8150_enabled(id);check(id>=0&&ci%4==0&&co%4==0&&w>0&&h>0,"TF8 layer params");if(!active){gpu.reset(ncnn::create_layer_vulkan("Convolution"));check(bool(gpu),"TF8 Vulkan fallback");return gpu->load_param(pd);}if(!env("SCV41_855_CALIBRATE").empty()){teacher.reset(ncnn::create_layer_vulkan("Convolution"));check(teacher&&!teacher->load_param(pd),"TF8 calibration teacher parameters");stats.assign(co*6,0);}return 0;}
 int load_model(const ncnn::ModelBin&mb)override {if(!active)return gpu->load_model(mb);auto weights=mb.load(weight_count,0);auto bias=mb.load(co,1);check(!weights.empty()&&!bias.empty(),"TF8 original weight stream");if(teacher){ncnn::Mat data[2]={weights,bias};ncnn::ModelBinFromMatArray original(data);check(!teacher->load_model(original),"TF8 teacher weights");}auto folder=env("SCV41_855_MODEL_ROOT","/var/tmp/scv41-855-mixed/models")+"/n_"+std::to_string(id);std::vector<float>values(2*ci+3*co);std::ifstream file(folder+"/bridge.f32",std::ios::binary);file.read(reinterpret_cast<char*>(values.data()),values.size()*4);check(bool(file),"TF8 scales");in_coeff=ncnn::Mat(2*ci);out_coeff=ncnn::Mat(2*co);std::memcpy(in_coeff.data,values.data(),ci*2*4);float*p=out_coeff;for(int c=0;c<co;++c){p[c]=values[2*ci+c];p[co+c]=values[2*ci+co+c];check(values[2*ci+2*co+c]==bias[c],"Centered source GPU bias provenance");}auto correction_dir=env("SCV41_855_CORRECTIONS");if(!correction_dir.empty()){std::vector<float>ab(co*2);std::ifstream correction(correction_dir+"/n_"+std::to_string(id)+".f32",std::ios::binary);correction.read(reinterpret_cast<char*>(ab.data()),ab.size()*4);check(bool(correction),"TF8 correction file");for(int c=0;c<co;++c){check(std::isfinite(ab[c])&&std::abs(ab[c]-1)<=.051f&&std::isfinite(ab[co+c]),"TF8 correction bounds");p[c]*=ab[c];p[co+c]=p[co+c]*ab[c]+ab[co+c];}}std::ifstream sparse_file(folder+"/outliers.csr",std::ios::binary);uint32_t header[3];sparse_file.read(reinterpret_cast<char*>(header),12);check(bool(sparse_file)&&header[0]==uint32_t(co)&&header[1]==uint32_t(ci)&&header[2]>0&&header[2]<=uint32_t(co*ci/20),"Sparse correction header");csr_nnz=header[2];csr=ncnn::Mat(co+1+2*csr_nnz+ci);sparse_file.read(reinterpret_cast<char*>(csr.data),csr.w*4);check(bool(sparse_file),"Sparse correction payload");const uint32_t*words=reinterpret_cast<const uint32_t*>(csr.data);check(words[0]==0&&words[co]==uint32_t(csr_nnz),"Sparse row endpoints");for(int c=0;c<co;++c)check(words[c]<=words[c+1],"Sparse row ordering");for(int j=0;j<csr_nnz;++j)check(words[co+1+j]<uint32_t(ci)&&std::isfinite(reinterpret_cast<const float*>(words+co+1+csr_nnz)[j]),"Sparse index or weight");sm8150_owner().call([&,this]{model=sm8150_get_model(folder,ci,co,w,h);});return 0;}
 int create_pipeline(const ncnn::Option&opt)override {if(!active){gpu->vkdev=vkdev;gpu->bottom_shapes=bottom_shapes;gpu->top_shapes=top_shapes;return gpu->create_pipeline(opt);}if(teacher){teacher->vkdev=vkdev;teacher->bottom_shapes=bottom_shapes;teacher->top_shapes=top_shapes;check(!teacher->create_pipeline(opt),"TF8 teacher pipeline");}auto f=fp32(opt);auto build=[&](const char*shader,std::unique_ptr<ncnn::Pipeline>&p){std::vector<uint32_t>spv;check(!ncnn::compile_spirv_module(shader,f,spv),"TF8 shader compile");p.reset(new ncnn::Pipeline(vkdev));p->set_subgroup_size(64);p->set_local_size_xyz(64,1,1);check(!p->create(spv.data(),spv.size()*4,{}),"TF8 pipeline create");};build(sm8150_quant_shader,quant);build(sm8150_dequant_shader,dequant);build(sm8150_quant_packed,packed_quant);build(sm8150_dequant_packed,packed_dequant);return 0;}
 int upload_model(ncnn::VkTransfer&cmd,const ncnn::Option&opt)override {if(!active)return gpu->upload_model(cmd,opt);if(teacher)check(!teacher->upload_model(cmd,opt),"TF8 teacher upload");auto f=fp32(opt);cmd.record_upload(in_coeff,in_gpu,f);cmd.record_upload(out_coeff,out_gpu,f);cmd.record_upload(csr,csr_gpu,f);return 0;}
 int destroy_pipeline(const ncnn::Option&opt)override {if(!active)return gpu->destroy_pipeline(opt);if(teacher)teacher->destroy_pipeline(opt);quant.reset();dequant.reset();packed_quant.reset();packed_dequant.reset();return 0;}
 int forward(const ncnn::VkMat&bottom,ncnn::VkMat&top,ncnn::VkCompute&cmd,const ncnn::Option&opt)const override {
  if(!active)return gpu->forward(bottom,top,cmd,opt);
  if(env("SCV41_855_PACKED_BRIDGE")=="1"&&!teacher&&bottom.elempack==4&&bottom.elembits()==16&&(opt.use_fp16_packed||opt.use_fp16_storage)){
   auto f=fp32(opt);check(bottom.w==w&&bottom.h==h&&bottom.c*4==ci,"Packed bridge input layout");
   ncnn::VkMat bytes(w*h*ci/4,size_t(4),f.blob_vkallocator),dispatch;dispatch.w=w*h*ci/4;dispatch.h=1;dispatch.c=1;std::vector<ncnn::vk_constant_type>p(5);p[0].i=w*h;p[1].i=ci;p[2].i=bottom.cstep;p[3].f=model->input_step;p[4].i=model->input_zero;
   cmd.record_pipeline(packed_quant.get(),{bottom,in_gpu,bytes},p,dispatch);ncnn::Mat host;cmd.record_clone(bytes,host,f);check(!cmd.submit_and_wait(),"Packed bridge input submit");cmd.reset();
   auto output=sm8150_owner().call([&]{return model->execute(host);});ncnn::Mat words(int(output.size()/4));std::memcpy(words.data,output.data(),output.size());ncnn::VkMat raw;cmd.record_clone(words,raw,f);
   top.create(w,h,co/4,size_t(8),4,opt.blob_vkallocator);check(!top.empty(),"Packed output allocation");p.resize(7);p[1].i=co;p[2].i=top.cstep;p[3].f=model->output_step;p[4].i=model->output_zero;p[5].i=bottom.cstep;p[6].i=csr_nnz;dispatch.w=w*h*co/2;cmd.record_pipeline(packed_dequant.get(),{raw,out_gpu,top,bottom,csr_gpu},p,dispatch);return 0;
  }
  auto f=fp32(opt);ncnn::VkMat gold;if(teacher)check(!teacher->forward(bottom,gold,cmd,opt),"TF8 teacher forward");ncnn::VkMat unpacked;vkdev->convert_packing(bottom,unpacked,1,1,cmd,f);check(unpacked.w==w&&unpacked.h==h&&unpacked.c==ci&&unpacked.elempack==1&&unpacked.elembits()==32,"TF8 GPU input layout");ncnn::VkMat bytes(w*h*ci/4,size_t(4),f.blob_vkallocator),dispatch;dispatch.w=w*h*ci/4;dispatch.h=1;dispatch.c=1;std::vector<ncnn::vk_constant_type>p(5);p[0].i=w*h;p[1].i=ci;p[2].i=unpacked.cstep;p[3].f=model->input_step;p[4].i=model->input_zero;cmd.record_pipeline(quant.get(),{unpacked,in_gpu,bytes},p,dispatch);ncnn::Mat host;cmd.record_clone(bytes,host,f);check(!cmd.submit_and_wait(),"TF8 input submit");cmd.reset();
  auto output=sm8150_owner().call([&]{return model->execute(host);});ncnn::Mat words(int(output.size()/4));std::memcpy(words.data,output.data(),output.size());ncnn::VkMat raw,restored(w,h,co,size_t(4),1,f.blob_vkallocator);cmd.record_clone(words,raw,f);p.resize(7);p[1].i=co;p[2].i=restored.cstep;p[3].f=model->output_step;p[4].i=model->output_zero;p[5].i=unpacked.cstep;p[6].i=csr_nnz;dispatch.w=w*h*co;cmd.record_pipeline(dequant.get(),{raw,out_gpu,restored,unpacked,csr_gpu},p,dispatch);
  if(teacher){ncnn::Mat x,y;cmd.record_download(restored,x,f);cmd.record_download(gold,y,f);check(!cmd.submit_and_wait(),"TF8 calibration submit");cmd.reset();check(x.c==co&&y.c==co&&x.elempack==1&&y.elempack==1,"TF8 calibration layout");for(int c=0;c<co;++c){const float*a=x.channel(c),*b=y.channel(c);double*t=stats.data()+c*6;for(int s=0;s<w*h;++s){double xx=a[s],yy=b[s];check(std::isfinite(xx)&&std::isfinite(yy),"TF8 calibration finite");t[0]+=xx;t[1]+=yy;t[2]+=xx*xx;t[3]+=yy*yy;t[4]+=xx*yy;}t[5]+=w*h;}top=gold;return 0;}
  vkdev->convert_packing(restored,top,4,(opt.use_fp16_packed||opt.use_fp16_storage)?2:1,cmd,opt);return 0;
 }
};
DEFINE_LAYER_CREATOR(NpuConv855)

int main(int argc,char**argv){try{
 const std::string mode=env("SCV41_PIPE_ENGINE","aip"),base=env("SCV41_PIPE_BASE","/var/tmp/scv41-ocr-pipeline/models");
 const bool npu=mode=="aip"||mode=="dsp",cpu=mode=="cpu";
 check(npu||cpu||mode=="gpu"||mode=="mixed","unknown engine");
 const int batch=integer("SCV41_PIPE_BATCH",1),workers=integer("SCV41_PIPE_WORKERS",1),capacity=integer("SCV41_PIPE_CONCURRENCY",10),limit=integer("SCV41_PIPE_LIMIT",40);
 check(batch>=1&&batch<=10&&workers>=1&&workers<=10&&capacity>=batch&&capacity<=10,"invalid scheduling config");
 std::vector<std::string>keys;std::string line;std::ifstream dict(base+"/keys.txt");while(std::getline(dict,line))keys.push_back(line);check(keys.size()==18710,"dictionary");
 std::vector<Row>rows;int source_index=0,offset=integer("SCV41_PIPE_OFFSET",0);std::ifstream list(env("SCV41_PIPE_INPUT_LIST",(base+"/all.tsv").c_str()));while(std::getline(list,line)&&int(rows.size())<limit){if(source_index++<offset)continue;auto tab=line.find('\t');check(tab!=std::string::npos,"list");rows.push_back({line.substr(0,tab),line.substr(tab+1)});}check(!rows.empty(),"empty list");
 auto init=now();std::unique_ptr<zdl::DlContainer::IDlContainer>container;std::unique_ptr<zdl::SNPE::SNPE>front;std::unique_ptr<zdl::DlSystem::ITensor>input;double step=1,zero=0,input_max=255;
 const std::string feature=env("SCV41_PIPE_FEATURE","t_25");int fc=128,fh=12,fw=160;
 if(feature=="t_293"){fc=768;fh=1;fw=80;}else if(feature=="t_4"){fc=64;fh=24;fw=320;}
 if(npu){
  const auto lib=env("SCV41_NPU_RPC_LIBRARY");check(!lib.empty(),"managed RPC owner required");auto*rpc=dlopen(lib.c_str(),RTLD_NOW|RTLD_GLOBAL);check(rpc,"RPC load");auto*paths=static_cast<char*>(dlsym(rpc,"DSP_LIBS_LOCATION"));check(paths,"RPC paths");const char*search="/opt/scv41-snpe-1.59.0.3230/dsp-original/cdsp;/opt/scv41-snpe-1.59.0.3230/lib/dsp;";std::memcpy(paths,search,std::strlen(search)+1);
  container=zdl::DlContainer::IDlContainer::open(env("SCV41_PIPE_DLC",(base+"/front-aip.dlc").c_str()).c_str());check(bool(container),"DLC");
  zdl::SNPE::SNPEBuilder b(container.get());b.setRuntimeProcessor(mode=="dsp"?zdl::DlSystem::Runtime_t::DSP_FIXED8_TF:zdl::DlSystem::Runtime_t::AIP_FIXED8_TF).setCPUFallbackMode(false).setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE);
  if(batch>1){zdl::DlSystem::TensorShapeMap shapes;shapes.add("t_0",zdl::DlSystem::TensorShape{size_t(batch),48,640,3});b.setInputDimensions(shapes);}
  front=b.build();check(bool(front),std::string("SNPE_BUILD: ")+zdl::DlSystem::getLastErrorString());
  auto dims=front->getInputDimensions("t_0");check(bool(dims),"input dims");input=zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*dims);check(input&&input->getSize()==size_t(batch)*92160,"batch dimensions rejected");
  auto outattrs=front->getInputOutputBufferAttributes(feature.c_str());check(bool(outattrs),"output attributes");auto odims=(*outattrs)->getDims();std::cout<<"FRONT_OUTPUT_DIMS";for(size_t d=0;d<odims.rank();++d)std::cout<<" "<<odims[d];std::cout<<std::endl;
  auto attrs=front->getInputOutputBufferAttributes("t_0");check(bool(attrs),"input attrs");auto*encoding=dynamic_cast<zdl::DlSystem::UserBufferEncodingTfN*>((*attrs)->getEncoding());check(encoding,"TF encoding");step=encoding->getQuantizedStepSize();zero=encoding->getStepExactly0();input_max=encoding->getElementSize()==2?65535:255;check(step>0&&zero<=input_max,"TF encoding range");std::cout<<"INPUT_ENCODING_BITS "<<encoding->getElementSize()*8<<std::endl;
  auto diagdir=env("SCV41_PIPE_DIAG");if(!diagdir.empty()){auto d=front->getDiagLogInterface();check(bool(d),"diagnostic interface");auto opt=(*d)->getOptions();opt.LogFileDirectory=diagdir;opt.DiagLogMask="DNN_RUNTIME=ON";check((*d)->setOptions(opt)&&(*d)->start(),"diagnostic start");}
 }
 if(!cpu)check(ncnn::get_gpu_count()==1,"GPU count");
 const double fallback_threshold=std::stod(env("SCV41_PIPE_FALLBACK_THRESHOLD","0"));
 const int ctc_mode=integer("SCV41_GPU_CTC_MODE",0);check(ctc_mode==0||!npu,"Fused CTC not qualified for NPU routing");
 std::vector<std::unique_ptr<FusedGpuCtc>>fused_ctc;
 std::vector<std::unique_ptr<ncnn::Net>>prefixes,tails,reference_cnns;
 for(int i=0;i<workers;++i){
  auto p=std::unique_ptr<ncnn::Net>(new ncnn::Net);if(mode=="mixed")p->register_custom_layer("NpuConv855",NpuConv855_layer_creator);p->opt.num_threads=integer("SCV41_PIPE_CPU_THREADS",1);p->opt.use_vulkan_compute=!cpu;p->opt.use_fp16_packed=!cpu;p->opt.use_fp16_storage=false;p->opt.use_fp16_arithmetic=!cpu&&std::getenv("SCV41_GPU_HALF_CNN");p->opt.use_fp16_uniform=false;p->opt.use_subgroup_ops=false;p->opt.use_cooperative_matrix=false;if(!cpu)p->set_vulkan_device(0);
  const auto path=cpu?base+"/cpu":npu?base+"/middle-"+feature:base+"/cnn";
  if(!npu||feature!="t_293")check(!p->load_param((path+"/tail.param").c_str())&&!p->load_model((path+"/tail.bin").c_str()),"prefix model "+path);
  auto t=std::unique_ptr<ncnn::Net>(new ncnn::Net);t->opt=p->opt;t->opt.use_fp16_packed=false;t->opt.use_fp16_arithmetic=false;if(!cpu){t->set_vulkan_device(0);check(!t->load_param((base+"/tail/tail.param").c_str())&&!t->load_model((base+"/tail/tail.bin").c_str()),"tail model");}auto r=std::unique_ptr<ncnn::Net>(new ncnn::Net);if(npu&&fallback_threshold>0){r->opt=p->opt;r->set_vulkan_device(0);check(!r->load_param((base+"/cnn/tail.param").c_str())&&!r->load_model((base+"/cnn/tail.bin").c_str()),"fallback CNN");}fused_ctc.emplace_back(!cpu&&ctc_mode?new FusedGpuCtc(t->opt):nullptr);reference_cnns.push_back(std::move(r));prefixes.push_back(std::move(p));tails.push_back(std::move(t));
 }
 std::cout<<"PIPE_INITIALIZED engine="<<mode<<" npu_contexts="<<(npu?1:sm8150_context_count.load())<<" workers="<<workers<<" batch="<<batch<<" concurrency="<<capacity<<" init_ms="<<ms(init)<<std::endl;
 std::ofstream output(env("SCV41_PIPE_OUTPUT"));check(bool(output),"output");
 std::mutex mutex,write_mutex;std::condition_variable cv;std::deque<Job>queue;bool done=false;std::atomic<bool>failed{false};size_t completed=0,admitted=0,peak=0;std::string error;std::vector<std::thread>threads;
 std::this_thread::sleep_for(std::chrono::seconds(6));double run=now();
 auto fail=[&](const std::string&s){std::lock_guard<std::mutex>lock(mutex);if(!failed){failed=true;error=s;}cv.notify_all();};
 for(int worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{try{while(true){Job job;{std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return failed||done||!queue.empty();});if(failed)return;if(queue.empty()&&done)return;job=std::move(queue.front());queue.pop_front();}
  auto gpu_start=now();ncnn::Mat y;bool use_fused=false;Decoded fused_decoded;
  if(cpu){auto ex=prefixes[worker]->create_extractor();check(!ex.input("t_0",job.x)&&!ex.extract("t_470",y),"CPU extract");}
  else{ncnn::Mat bridge=job.x;if(!npu||feature!="t_293"){auto ex=prefixes[worker]->create_extractor();check(!ex.input(npu?feature.c_str():"t_0",job.x)&&!ex.extract("t_293",bridge),"CNN extract");}auto ex=tails[worker]->create_extractor();check(!ex.input("t_293",bridge),"tail input");if(ctc_mode){fused_decoded=fused_ctc[worker]->execute(ex,keys,ctc_mode==2);use_fused=true;}else check(!ex.extract("t_470",y),"tail extract");}
  double gpu_ms=ms(gpu_start);if(!use_fused)check(y.elempack==1&&y.elembits()==32&&y.w==18710&&y.h==80&&y.c==1,"output shape");
  auto decode=now();auto decoded=use_fused?fused_decoded:decode_output(y,keys);double decode_ms=ms(decode),confidence=decoded.confidence;bool fallback=npu&&fallback_threshold>0&&confidence<fallback_threshold;double fallback_ms=0;
  if(fallback){auto t=now();std::vector<float>raw(92160);std::ifstream f(job.row.path,std::ios::binary);f.read(reinterpret_cast<char*>(raw.data()),92160*4);check(bool(f),"fallback input");ncnn::Mat x(640,48,3);for(int k=0;k<3;++k){float*p=x.channel(k);for(int z=0;z<48*640;++z)p[z]=raw[z*3+k];}auto ex=reference_cnns[worker]->create_extractor();ncnn::Mat bridge;check(!ex.input("t_0",x)&&!ex.extract("t_293",bridge),"fallback CNN execute");auto tail=tails[worker]->create_extractor();check(!tail.input("t_293",bridge)&&!tail.extract("t_470",y),"fallback tail execute");decoded=decode_output(y,keys);fallback_ms=ms(t);}
  double end=now();{std::lock_guard<std::mutex>lock(write_mutex);output<<std::setprecision(14)<<"{\"id\":"<<quote(job.row.id)<<",\"text\":"<<quote(decoded.text)<<",\"hash\":\""<<std::hex<<decoded.hash<<std::dec<<"\",\"confidence\":"<<confidence<<",\"fallback\":"<<(fallback?"true":"false")<<",\"fallback_ms\":"<<fallback_ms<<",\"worker\":"<<worker<<",\"start_s\":"<<job.start<<",\"end_s\":"<<end<<",\"front_start_s\":"<<job.front_start<<",\"front_end_s\":"<<job.front_end<<",\"gpu_start_s\":"<<gpu_start<<",\"gpu_end_s\":"<<decode<<",\"front_ms\":"<<job.front<<",\"bridge_ms\":"<<job.bridge<<",\"gpu_ms\":"<<gpu_ms<<",\"queue_ms\":"<<(gpu_start-job.ready)*1000<<",\"decode_ms\":"<<decode_ms<<",\"total_ms\":"<<(end-job.start)*1000<<"}\n";output.flush();check(bool(output),"write");}
  {std::lock_guard<std::mutex>lock(mutex);++completed;cv.notify_all();}
 }}catch(const std::exception&e){fail(e.what());}});
 try{for(size_t index=0;index<rows.size();){size_t count=std::min(size_t(batch),rows.size()-index);double start;
  {std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return failed||admitted-completed+count<=size_t(capacity);});check(!failed,"consumer failed");start=now();admitted+=count;peak=std::max(peak,admitted-completed);}
  std::vector<Job>jobs(count);std::vector<std::vector<float>>raw(count,std::vector<float>(92160));for(size_t j=0;j<count;++j){jobs[j].row=rows[index+j];jobs[j].start=start;std::ifstream f(jobs[j].row.path,std::ios::binary);f.read(reinterpret_cast<char*>(raw[j].data()),92160*4);check(bool(f),"input file");}
  double front_ms=0,bridge_ms=0,front_start=0,front_end=0;
  if(npu){auto it=input->begin();for(int b=0;b<batch;++b)for(size_t k=0;k<92160;++k,++it){double v=b<int(count)?raw[b][k]:0;double q=std::nearbyint(v/step+zero);*it=(std::fmin(input_max,std::fmax(0.,q))-zero)*step;}
   zdl::DlSystem::TensorMap in,out;in.add("t_0",input.get());auto t=now();front_start=t;check(front->execute(in,out),std::string("SNPE_EXECUTE: ")+zdl::DlSystem::getLastErrorString());front_end=now();front_ms=(front_end-t)*1000;auto*y=out.getTensor(feature.c_str());check(y&&y->getSize()==size_t(batch)*fc*fh*fw,"front output dims");t=now();std::vector<float>values(y->begin(),y->end());if(index==0&&!env("SCV41_PIPE_DUMP").empty()){std::ofstream dump(env("SCV41_PIPE_DUMP"),std::ios::binary);dump.write(reinterpret_cast<const char*>(values.data()),values.size()*4);}for(size_t j=0;j<count;++j){jobs[j].x=ncnn::Mat(fw,fh,fc);for(int k=0;k<fc;++k){float*p=jobs[j].x.channel(k);for(int z=0;z<fh*fw;++z)p[z]=values[(j*fh*fw+z)*fc+k];}}bridge_ms=ms(t);
  }else{for(size_t j=0;j<count;++j){jobs[j].x=ncnn::Mat(640,48,3);for(int k=0;k<3;++k){float*p=jobs[j].x.channel(k);for(int z=0;z<48*640;++z)p[z]=raw[j][z*3+k];}}}
  {std::lock_guard<std::mutex>lock(mutex);for(auto&job:jobs){job.front=front_ms;job.front_start=front_start;job.front_end=front_end;job.bridge=bridge_ms;job.ready=now();queue.push_back(std::move(job));}cv.notify_all();}index+=count;
 }}catch(const std::exception&e){fail(e.what());}
 {std::lock_guard<std::mutex>lock(mutex);done=true;cv.notify_all();}for(auto&t:threads)t.join();check(!failed,"PIPE_FAILED "+error);check(completed==rows.size(),"incomplete");
 if(front&&!env("SCV41_PIPE_DIAG").empty())(*front->getDiagLogInterface())->stop();
 std::cout<<"SM8150_NPU_LINEAR_EXECUTIONS "<<sm8150_executions.load()<<" snpe_execute_us="<<sm8150_execute_us.load()<<std::endl;std::cout<<"PIPE_COMPLETE "<<completed<<" duration_ms="<<ms(run)<<" peak_inflight="<<peak<<std::endl;std::this_thread::sleep_for(std::chrono::seconds(6));return 0;
}catch(const std::exception&e){std::cerr<<"PIPE_ERROR "<<e.what()<<std::endl;return 70;}}
