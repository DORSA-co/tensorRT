import pycuda.autoinit
import pycuda.driver as cuda

def meminfo(kernel):
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    mbpt=kernel.max_threads_per_block
    print("=MEM=\nLocal:%d,\nShared:%d,\nRegisters:%d,\nConst:%d,\nMax Threads/B:%d" % (local,shared,regs,const,mbpt))

(free,total)=cuda.mem_get_info()
print("Global memory occupancy:%f%% free"%(free*100/total))

for devicenum in range(cuda.Device.count()):
    device=cuda.Device(devicenum)
    attrs=device.get_attributes()

    #Beyond this point is just pretty printing
    print("\n===Attributes for device %d"%devicenum)
    for (key,value) in attrs.items():
        print("%s:%s"%(str(key),str(value)))
