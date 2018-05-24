from __future__ import division
from __future__ import print_function
import os
import time
import math
import numpy as np
import pyopencl as cl


class CLWrapper:
	"class holds information about OpenCL state"

	def __init__(self, batchSize, maxT, maxC, kernelVariant=1, enableGPUDebug=False):
		"specify size: number of batch elements, number of time-steps, number of characters. Set kernelVariant to either 1 or 2. Set enableGPUDebug to True to debug kernel via CodeXL."

		# force rebuild of program such that GPU debugger can attach to kernel
		self.enableGPUDebug = enableGPUDebug
		if enableGPUDebug:
			os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
			os.environ['PYOPENCL_NO_CACHE'] = '1'

		#consts
		self.batchSize = batchSize
		self.maxT = maxT
		self.maxC = maxC
		assert kernelVariant in [1, 2]
		self.kernelVariant = kernelVariant

		# platform, context, queue
		platforms = cl.get_platforms()
		assert platforms
		self.platform = platforms[0] # take first platform
		devices = self.platform.get_devices(cl.device_type.GPU) # get GPU devices
		assert devices
		self.device = devices[0] # take first GPU
		self.context = cl.Context([self.device]) # context contains the first GPU
		self.queue = cl.CommandQueue(self.context, self.device) # command queue to first GPU

		# buffer
		sizeOfFloat32 = 4
		batchBufSize = batchSize * maxC * maxT * sizeOfFloat32
		self.batchBuf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=batchBufSize, hostbuf=None)
		self.res = np.zeros([batchSize, maxT]).astype(np.int32)
		self.resBuf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.res.nbytes)
		self.tmpBuf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.res.nbytes)

		# compile program and use defines for program-constants to avoid passing private variables
		buildOptions = '-D STEP_BEGIN={} -D MAX_T={} -D MAX_C={}'.format(2 ** math.ceil(math.log2(maxT)), maxT, maxC)
		self.program = cl.Program(self.context, open('BestPathCL.cl').read()).build(buildOptions)

		# variant 1: single pass
		if kernelVariant == 1:
			self.kernel1 = cl.Kernel(self.program, 'bestPathAndCollapse')
			self.kernel1.set_arg(0, self.batchBuf)
			self.kernel1.set_arg(1, self.resBuf)

			# all time-steps must fit into a work-group
			assert maxT <= self.kernel1.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device)

		# variant 2: two passes
		else:
			# kernel1: calculate best path
			self.kernel1 = cl.Kernel(self.program, 'bestPath')
			self.kernel1.set_arg(0, self.batchBuf)
			self.kernel1.set_arg(1, self.tmpBuf)

			# kernel2: collapse best path
			self.kernel2 = cl.Kernel(self.program, 'collapsePath')
			self.kernel2.set_arg(0, self.tmpBuf)
			self.kernel2.set_arg(1, self.resBuf)

			# all chars must fit into a work-group
			assert maxC <= self.kernel1.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device)


	def compute(self, batch):
		"compute best path for each batch element. Returns blank-terminated label strings for batch elements."

		# measure time in GPU debug mode
		if self.enableGPUDebug:
			t0 = time.time()

		# copy batch to device
		cl.enqueue_write_buffer(self.queue, self.batchBuf, batch.astype(np.float32), is_blocking=False)

		# one pass
		if self.kernelVariant == 1:
			cl.enqueue_nd_range_kernel(self.queue, self.kernel1, (self.batchSize, self.maxT), (1, self.maxT))
		# two passes
		else:
			cl.enqueue_nd_range_kernel(self.queue, self.kernel1, (self.batchSize, self.maxT, self.maxC), (1, 1, self.maxC))
			cl.enqueue_nd_range_kernel(self.queue, self.kernel2, (self.batchSize,), None)

		# copy result back from GPU and return it
		cl.enqueue_read_buffer(self.queue, self.resBuf, self.res, is_blocking=True)

		# measure time in GPU debug mode
		if self.enableGPUDebug:
			t1 = time.time()
			print('BestPathCL.compute(...) time: ', t1-t0)

		return self.res


def ctcBestPathCL(batch, classes, clWrapper):
	"implements best path decoding on the GPU with OpenCL"

	# compute best labeling
	labelStrBatch = clWrapper.compute(batch)

	#go over batch
	blank = len(classes)
	charStrBatch = []
	for b in range(clWrapper.batchSize):
		# map to chars
		charStr = ''
		for label in labelStrBatch[b]:
			if label == blank:
				break
			charStr += classes[label]
		charStrBatch.append(charStr)

	return charStrBatch


def testBestPathCL():
	"test decoder"
	classes = 'ab'
	mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
	maxT, maxC = mat.shape
	clWrapper = CLWrapper(1, maxT, maxC, enableGPUDebug=True)
	print('Test best path decoding (CL)')
	expected = ''
	actual = ctcBestPathCL(np.stack([mat]), classes, clWrapper)[0]
	print('Expected: "' + expected + '"')
	print('Actual: "' + actual + '"')
	print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
	testBestPathCL()
