#!/usr/bin/python3
"""Run the fixed SCV41 AIP model; retain the original firmware power state."""
import fcntl
import os
import signal
import struct
import subprocess
import sys

if os.geteuid() != 0:
    sys.exit('Run with sudo through scv41-npu-check.')

base = '/opt/scv41-snpe-1.59.0.3230'
rpc = base + '/fastrpc-v4'
os.makedirs('/run/scv41-npu-runtime',mode=0o700,exist_ok=True)
lock_fd=os.open('/run/scv41-npu-runtime/lock',os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o600)
try:
    fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit('NPU_TEST_BUSY: another managed NPU client owns the runtime')
fd = os.open('/dev/msm_npu', os.O_RDWR | os.O_CLOEXEC)

def state():
    data = bytearray(struct.pack('<11I', 0x100, 1, 0, *([0] * 8)))
    fcntl.ioctl(fd, 0x402c6e0b, data, True)
    return struct.unpack('<11I', data)[3]

def power(value):
    fcntl.ioctl(fd, 0x402c6e0a,
                struct.pack('<11I', 0x100, 1, 0, value, *([0] * 7)))

def interrupted(number, frame):
    raise SystemExit(128 + number)

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
before = state()
enabled = False
child = None
print('NPU_FIRMWARE_STATE_BEFORE=', before, flush=True)
try:
    if before == 0:
        power(1)
        enabled = True
    if state() != 1:
        raise RuntimeError('NPU firmware did not become ready')
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    if os.environ.get('SCV41_NPU_CACHE_SYNC')=='1':env['LD_PRELOAD']='/opt/scv41-npu-test/libnpu-cache-sync.so'
    if os.environ.get('SCV41_NPU_UNCACHED_ION','1')=='1':
        helper='/opt/scv41-npu-test/libnpu-uncached-ion.so'
        if not os.path.isfile(helper):raise RuntimeError('Required qualified NPU ION helper is missing')
        env['LD_PRELOAD']=helper
    env.update({
        'LD_LIBRARY_PATH': rpc + ':' + base + '/compat:/usr/lib/aarch64-linux-gnu/android:' + os.environ.get('SCV41_SNPE_NATIVE_LIBRARY_DIR',base+'/lib/aarch64-oe-linux-gcc9.3')+':'+base+'/lib/aarch64-oe-linux-gcc9.3',
        'SCV41_NPU_RPC_LIBRARY': rpc + '/libcdsprpc.so',
        'ADSP_LIBRARY_PATH': base + '/lib/dsp;' + base + '/dsp-original/cdsp;/lib/firmware/scv41;',
    })
    deadline=int(os.environ.get('SCV41_NPU_TEST_TIMEOUT','25'))
    if not 1<=deadline<=170:raise ValueError('NPU test timeout outside bounded range')
    binary=os.environ.get('SCV41_NPU_TEST_BINARY','/opt/scv41-npu-test/simple-npu-test')
    full_suite=binary=='/opt/scv41-npu-test/simple-npu-test' and 'SCV41_NPU_KNOWN_MODEL' not in env
    fixtures=('affine','mixed-sign') if full_suite else ('custom',)
    result=1
    for fixture in fixtures:
        child_env=env.copy()
        if full_suite:
            child_env['SCV41_NPU_NUMERIC_REPS']='12'
            child_env.pop('SCV41_NPU_MIXED_SIGN',None)
            child_env['SCV41_NPU_KNOWN_MODEL']='/opt/scv41-npu-test/models/known-affine-aip.dlc'
            if fixture=='mixed-sign':
                child_env['SCV41_NPU_KNOWN_MODEL']='/opt/scv41-npu-test/models/known-signed-aip.dlc'
                child_env['SCV41_NPU_MIXED_SIGN']='1'
        command=[binary,os.environ.get('SCV41_NPU_RUNTIME','aip')]
        if os.environ.get('SCV41_NPU_TRACE_IOCTLS')=='1':command=['strace','-f','-qq','-yy','-e','trace=ioctl','-o','/var/tmp/scv41-npu-current-ioctl.txt']+command
        child=subprocess.Popen(command,env=child_env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            output,_=child.communicate(timeout=deadline)
            sys.stdout.buffer.write(output);sys.stdout.flush();result=child.returncode
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);output,_=child.communicate()
            sys.stdout.buffer.write(output);sys.stdout.flush();result=124
            print('NPU_TEST_TIMEOUT',flush=True)
        if full_suite and result==0:
            marker=('NPU_NUMERIC_SUITE_PASS runtime=aip repetitions=12 fixture='+fixture).encode()
            if marker not in output:
                result=65;print('NPU_TEST_INCOMPLETE_SUITE',flush=True)
        if result!=0:break
    if full_suite and result==0:print('SCV41_NPU_FULL_NUMERIC_PASS fixtures=2 repetitions_per_fixture=12',flush=True)
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGKILL)
        child.wait()
    if enabled:
        power(0)
    print('NPU_FIRMWARE_STATE_RESTORED=', state(), flush=True)
    os.close(fd)
    os.close(lock_fd)
sys.exit(result)
