/* 
best path decoding implemented in OpenCL
two variants are provided:
	variant 1: single-pass kernel
	variant 2: two-pass kernel
time measurement for 10000 batch elements on an AMD Radeon 8570: 
	variant 1: 25ms
	variant 2: 355ms + 5ms
MAX_T, MAX_C and STEP_BEGIN are defined via program build options to avoid passing constant values to each kernel
 */


// index of BxTxC matrix to 1d index
int btcOffset1d(int b, int t)
{
	return b * MAX_C * MAX_T + t * MAX_C;
}


// index of BxT matrix to 1d index
int btOffset1d(int b)
{
	return b * MAX_T;
}


// variant 1: single pass kernel
__kernel void bestPathAndCollapse(__global float* in, __global int* out)
{
	// constants
	const int b = get_global_id(0);
	const int t = get_local_id(1);
	const int blankLabel = MAX_C - 1;
	const int btcOffset = btcOffset1d(b, t);
	
	// find character with highest probability
	float bestVal = 0.0f;
	int bestIdx = 0;
	for(int c = 0; c < MAX_C; ++c)
	{
		const float currVal = in[btcOffset + c];
		if(currVal > bestVal)
		{
			bestVal = currVal;
			bestIdx = c;
		}
	}
	
	// save result in local memory
	__local int locIdx[MAX_T];
	locIdx[t] = bestIdx;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// collapse
	if(t == 0)
	{
		const int btOffset = btOffset1d(b);
		int lastLabel = blankLabel;
		int v = 0;
		for(int u = 0; u < MAX_T; ++u)
		{
			const int currLabel = locIdx[u];
			if(currLabel != lastLabel && currLabel != blankLabel)
			{
					out[btOffset + v] = currLabel;
					v++;
			}
			lastLabel = currLabel;
		}
		
		// put end marker at end of label string if needed
		if(v != MAX_T)
		{
			out[btOffset + v] = blankLabel;
		}
	}
}


// struct holds index and value of a character
typedef struct __attribute__ ((packed))
{
	float val;
	int idx;
} ValueIndexPair;


// variant 2: pass 1/2, compute best path
__kernel void bestPath(__global float* in, __global int* out)
{
	// constants
	const int b = get_global_id(0);
	const int t = get_global_id(1);
	const int c = get_local_id(2);
	
	// put into local memory
	__local ValueIndexPair valueIndexPairs[MAX_C];
	__local ValueIndexPair* currPtr = valueIndexPairs + c;
	currPtr->val = in[btcOffset1d(b, t)+c];
	currPtr->idx = c;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// reduce to largest value and corresponding index
	for(int i = STEP_BEGIN; i > 0; i >>= 1)
	{
		if(c < i && c + i < MAX_C)
		{
			__local ValueIndexPair* otherPtr = valueIndexPairs + c + i;
			*currPtr = currPtr->val < otherPtr->val ? *otherPtr : *currPtr;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// write best label index to global memory
	if(c == 0)
	{
		out[btOffset1d(b) + t] = currPtr->idx;
	}
}


// variant 2: pass 2/2, collapse best path
__kernel void collapsePath(__global int* in, __global int* out)
{
	// constants
	const int b = get_global_id(0);
	const int blankLabel = MAX_C - 1;
	const int btOffset = btOffset1d(b);
	
	// collapse
	int lastLabel = blankLabel;
	int v = 0;
	for(int u = 0; u < MAX_T; ++u)
	{
		const int currLabel = in[btOffset + u];
		if(currLabel != lastLabel && currLabel != blankLabel)
		{
				out[btOffset + v] = currLabel;
				v++;
		}
		lastLabel = currLabel;
	}
	
	// put end marker at end of label string if needed
	if(v != MAX_T)
	{
		out[btOffset + v] = blankLabel;
	}
}

