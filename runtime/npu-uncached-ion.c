#define _GNU_SOURCE
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
struct ion_allocation { uint64_t length; uint32_t heap_mask,flags,fd,unused; };
_Static_assert(sizeof(struct ion_allocation)==24,"SCV41 fd-returning ION ABI");
static void barrier(void) {
#if defined(__aarch64__)
 __asm__ volatile("dsb sy" ::: "memory");
#else
 __sync_synchronize();
#endif
}
int ioctl(int fd,unsigned long request,...) {
 va_list ap;va_start(ap,request);void* arg=va_arg(ap,void*);va_end(ap);
 int allocation=_IOC_TYPE(request)=='I' && _IOC_NR(request)==0 && _IOC_SIZE(request)==24 && arg;
 uint32_t before=0;
 if(allocation) {
  struct ion_allocation* data=arg;before=data->flags;
  if(before&1)data->flags&=~3U;
 }
 barrier();int result=syscall(SYS_ioctl,fd,request,arg);int saved=errno;barrier();
 if(allocation) {
  struct ion_allocation* data=arg;
  fprintf(stderr,"NPU_ION_UNCACHED length=%llu heap=%x flags_before=%x flags_after=%x result=%d\n",(unsigned long long)data->length,data->heap_mask,before,data->flags,result);
 }
 errno=saved;return result;
}
