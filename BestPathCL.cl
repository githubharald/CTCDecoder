// index of BxTxC matrix to 1d index
int BTCIndex1d(int b, int t, int c, int maxT, int maxC)
{
	return b * maxC * maxT + t * maxC + c;
}


// index of BxT matrix to 1d index
int BTIndex1d(int b, int t, int maxT)
{
	return b * maxT + t;
}


// round integer value to next largest power of 2 (if not yet power of 2), e.g. 100->128 or 256->256
int roundUpPow2(int val)
{
	// taken from: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	val--;
	val |= val >> 1;
	val |= val >> 2;
	val |= val >> 4;
	val |= val >> 8;
	val |= val >> 16;
	val++;
	
	return val;
}


// variant 1: single pass kernel
__kernel void bestPathAndCollapse(__global float* in, int maxT, int maxC, __local int* locIdx, __global int* out)
{
	// constants
	const int b = get_global_id(0);
	const int t = get_local_id(1);
	const int blankLabel = maxC - 1;
	
	float bestVal = 0.0f;
	int bestIdx = 0;
	for(int c = 0; c < maxC; ++c)
	{
		const float currVal = in[BTCIndex1d(b, t, c, maxT, maxC)];
		if(currVal > bestVal)
		{
			bestVal = currVal;
			bestIdx = c;
		}
	}
	locIdx[t] = bestIdx;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// collapse
	if(t == 0)
	{
		int lastLabel = blankLabel;
		int currLabel = blankLabel;
		int v = 0;
		for(int u = 0; u < maxT; ++u)
		{
			currLabel = locIdx[u];
			if(currLabel != lastLabel && currLabel != blankLabel)
			{
					out[BTIndex1d(b, v, maxT)] = currLabel;
					v++;
			}
			lastLabel = currLabel;
		}
		
		// put end marker at end of label string if needed
		if(v != maxT)
		{
			out[BTIndex1d(b, v, maxT)] = blankLabel;
		}
	}
}


// variant 2: pass 1/2, compute best path
__kernel void bestPath(__global float* in, int maxT, int maxC, __local float* locVal, __local int* locIdx, __global int* out)
{
	// constants
	const int b = get_global_id(0);
	const int t = get_global_id(1);
	const int c = get_local_id(2);
	
	// put into local memory
	locVal[c] = in[BTCIndex1d(b, t, c, maxT, maxC)];
	locIdx[c] = c;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// reduce to largest value and corresponding index
	for(int i = roundUpPow2(maxC) / 2; i > 0; i >>= 1)
	{
		if(c < i)
		{
			const int d = c + i >= maxC ? c + i - maxC : c + i; // faster than (c + i) % maxC
			if(locVal[c] < locVal[d])
			{
				locVal[c] = locVal[d];
				locIdx[c] = locIdx[d];
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// write best label index to global memory
	if(c == 0)
	{
		out[BTIndex1d(b, t, maxT)] = locIdx[0];
	}
}


// variant 2: pass 2/2, collapse best path
__kernel void collapsePath(__global int* in, int maxT, int maxC, __global int* out)
{
	// constants
	const int b = get_global_id(0);
	const int blankLabel = maxC - 1;
	
	// collapse
	int lastLabel = blankLabel;
	int currLabel = blankLabel;
	int v = 0;
	for(int u = 0; u < maxT; ++u)
	{
		currLabel = in[BTIndex1d(b, u, maxT)];
		if(currLabel != lastLabel && currLabel != blankLabel)
		{
				out[BTIndex1d(b, v, maxT)] = currLabel;
				v++;
		}
		lastLabel = currLabel;
	}
	
	// put end marker at end of label string if needed
	if(v != maxT)
	{
		out[BTIndex1d(b, v, maxT)] = blankLabel;
	}
	
}

