#include "host_driver.hpp"
#include <climits>
namespace kdhost {
static_assert(sizeof(kdcnn::Result)==24,"Result ABI must remain six32-bit words");
void Driver::set(uint32_t r,uint32_t value){io.write32(io.context,control+r,value);}
void Driver::pointer(uint32_t r,uint64_t value){set(r,uint32_t(value));set(r+4,uint32_t(value>>32));}
Decision Driver::run(const uint8_t *model,uint32_t bytes,const uint16_t *ids,const uint8_t *masks,
 uint32_t windows,int32_t threshold,const uint8_t *raw,size_t rawbytes,uint32_t polls){
 Decision result;result.cost.cold=!loaded;
 if(faulted){result.error=10;return result;}
 if(bytes!=kdcnn::PARAM_BYTES||!model||!ids||!masks||!windows||!polls||(!raw&&rawbytes)||uint64_t(windows)*kdcnn::L>SIZE_MAX/2){result.error=11;return result;}
 for(uint64_t i=0;i<uint64_t(windows)*kdcnn::L;++i)if(masks[i]>1||(masks[i]&&ids[i]>=kdcnn::V)){result.error=12;return result;}
 if(!loaded){
  if(!io.write_memory(io.context,buffers.model,model,bytes)){result.error=13;faulted=true;return result;}
  io.flush(io.context,buffers.model,bytes);result.cost.model_write_bytes+=bytes;
 }
 int32_t previous=INT_MIN;uint32_t done=0;
 while(done<windows){
  const uint32_t count=windows-done>kdcnn::MAX_WINDOWS?kdcnn::MAX_WINDOWS:windows-done;
  const size_t idbytes=count*kdcnn::L*sizeof(uint16_t),maskbytes=count*kdcnn::L;
  if(!io.write_memory(io.context,buffers.ids,ids+size_t(done)*kdcnn::L,idbytes)||!io.write_memory(io.context,buffers.masks,masks+size_t(done)*kdcnn::L,maskbytes)){result.error=13;faulted=true;return result;}
  result.cost.input_write_bytes+=idbytes+maskbytes;
  io.flush(io.context,buffers.ids,idbytes);io.flush(io.context,buffers.masks,maskbytes);
  // A stale dirty CPU line must not overwrite the device's later result.
  io.flush(io.context,buffers.output,sizeof(kdcnn::Result));
  pointer(regs.blob,buffers.model);set(regs.bytes,bytes);pointer(regs.ids,buffers.ids);pointer(regs.masks,buffers.masks);
  set(regs.count,count);set(regs.load,!loaded);set(regs.first,done==0);set(regs.final,done+count==windows);
  set(regs.previous,uint32_t(previous));set(regs.previous_windows,done);set(regs.threshold,uint32_t(threshold));pointer(regs.output,buffers.output);
  io.barrier(io.context);set(regs.control,1);bool completed=false;
  for(uint32_t i=0;i<polls;++i){++result.cost.polls;if(io.read32(io.context,control+regs.control)&2){completed=true;break;}}
  if(!completed){result.error=14;faulted=true;loaded=false;return result;}
  io.barrier(io.context);io.invalidate(io.context,buffers.output,sizeof(kdcnn::Result));
  kdcnn::Result value{};
  if(!io.read_memory(io.context,buffers.output,&value,sizeof(value))){result.error=13;faulted=true;loaded=false;return result;}
  result.cost.output_read_bytes+=sizeof(value);++result.cost.chunks;
  const bool final=done+count==windows;
  if(value.error||value.windows!=done+count||value.decision_valid!=uint32_t(final)||value.blocked!=int(final&&value.maximum>threshold)||value.maximum<previous){result.error=15;faulted=true;loaded=false;return result;}
  loaded=true;previous=value.maximum;done+=count;result.cost.windows=done;
 }
 result.margin=previous;result.blocked=previous>threshold;result.valid=true;
 io.release_document(io.context,raw,rawbytes,result.blocked);return result;
}
}
