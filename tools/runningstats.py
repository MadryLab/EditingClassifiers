'''
Running statistics on the GPU using pytorch, by David Bau.

RunningTopK maintains top-k statistics for a set of channels in parallel.
RunningQuantile maintains (sampled) quantile statistics for a set of channels.
RunningVariance calculate running mean and variance statistics stably.
RunningCovariance and RunningCrossCovariance accumulate covariance statistics.
RunningSecondMoment adds up 2nd moment (covariance without subtracting mean).
RunningBincount does a running sparse count.
RunningAllIntersectionAndUnion count up intersection and unions.
RunningConditional[stats] keeps many running stats, each conditioned on a key.

Batchwise tally functions, analogous to tensor.topk, mean+variance,
bincount, covaraince, and sort (for quantiles), implemented in a way
that permits fast computation of statistics over large data sets that
do not fit in memory at once.

These functions are useful because, while many statistics are much
cheaper to compute on the GPU than on the CPU, they may require too
much memory to compute all at once.  Instead the statistics need
to be computed in a running fashion, one batch at a time, and
accumulated in a way that economizes GPU memory.
'''

import torch
import math
import numpy
from collections import defaultdict


class RunningTopK:
    '''
    A class to keep a running tally of the the top k values (and indexes)
    of any number of torch feature components.  Will work on the GPU if
    the data is on the GPU.

    This version flattens all arrays to avoid crashes.
    '''

    def __init__(self, k=100, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.k = k
        self.count = 0
        # This version flattens all data internally to 2-d tensors,
        # to avoid crashes with the current pytorch topk implementation.
        # The data is puffed back out to arbitrary tensor shapes on ouput.
        self.data_shape = None
        self.top_data = None
        self.top_index = None
        self.next = 0
        self.linear_index = 0
        self.perm = None

    def add(self, data, index=None):
        '''
        Adds a batch of data to be considered for the running top k.
        The zeroth dimension enumerates the observations.  All other
        dimensions enumerate different features.
        '''
        if self.top_data is None:
            # Allocation: allocate a buffer of size 5*k, at least 10, for each.
            self.data_shape = data.shape[1:]
            feature_size = int(numpy.prod(self.data_shape))
            self.top_data = torch.zeros(
                feature_size, max(10, self.k * 5), out=data.new())
            self.top_index = self.top_data.clone().long()
            self.linear_index = 0 if len(data.shape) == 1 else torch.arange(
                feature_size, out=self.top_index.new()).mul_(
                self.top_data.shape[-1])[:, None]
        size = data.shape[0]
        sk = min(size, self.k)
        if self.top_data.shape[-1] < self.next + sk:
            # Compression: if full, keep topk only.
            self.top_data[:, :self.k], self.top_index[:, :self.k] = (
                self.result(sorted=False, flat=True))
            self.next = self.k
            free = self.top_data.shape[-1] - self.next
        # Pick: copy the top sk of the next batch into the buffer.
        # Currently strided topk is slow.  So we clone after transpose.
        # TODO: remove the clone() if it becomes faster.
        cdata = data.reshape(size, -1).t().clone()
        td, ti = cdata.topk(sk, sorted=False)
        self.top_data[:, self.next:self.next + sk] = td
        if index is not None:
            ti = index[ti]
        else:
            ti = ti + self.count
        self.top_index[:, self.next:self.next + sk] = ti
        self.next += sk
        self.count += size

    def size(self):
        return self.count

    def result(self, sorted=True, flat=False):
        '''
        Returns top k data items and indexes in each dimension,
        with channels in the first dimension and k in the last dimension.
        '''
        k = min(self.k, self.next)
        # bti are top indexes relative to buffer array.
        td, bti = self.top_data[:, :self.next].topk(k, sorted=sorted)
        # we want to report top indexes globally, which is ti.
        ti = self.top_index.view(-1)[
            (bti + self.linear_index).view(-1)
        ].view(*bti.shape)
        if flat:
            return td, ti
        else:
            return (td.view(*(self.data_shape + (-1,))),
                    ti.view(*(self.data_shape + (-1,))))

    def to_(self, device):
        self.top_data = self.top_data.to(device)
        self.top_index = self.top_index.to(device)
        if isinstance(self.linear_index, torch.Tensor):
            self.linear_index = self.linear_index.to(device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            k=self.k,
            count=self.count,
            data_shape=tuple(self.data_shape),
            top_data=self.top_data.cpu().detach().numpy(),
            top_index=self.top_index.cpu().detach().numpy(),
            next=self.next,
            linear_index=(self.linear_index.cpu().numpy()
                          if isinstance(self.linear_index, torch.Tensor)
                          else self.linear_index),
            perm=self.perm)

    def set_state_dict(self, dic):
        self.k = dic['k'].item()
        self.count = dic['count'].item()
        self.data_shape = tuple(dic['data_shape'])
        self.top_data = torch.from_numpy(dic['top_data'])
        self.top_index = torch.from_numpy(dic['top_index'])
        self.next = dic['next'].item()
        self.linear_index = (torch.from_numpy(dic['linear_index'])
                             if len(dic['linear_index'].shape) > 0
                             else dic['linear_index'].item())


class RunningConditionalTopK:
    def __init__(self, k=None, state=None):
        self.running_topk = {}
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.k = k
        self.count = 0

    def add(self, condition, data, index):
        if condition not in self.running_topk:
            self.running_topk[condition] = RunningTopK()
        rv = self.running_topk[condition]
        rv.add(data, index)
        self.count += len(data)

    def keys(self):
        return self.running_topk.keys()

    def conditional(self, c):
        return self.running_topk[c]

    def has_conditional(self, c):
        return c in self.running_topk

    def to_(self, device, conditions=None):
        if conditions is None:
            conditions = self.keys()
        for cond in conditions:
            if cond in self.running_topk:
                self.running_topk[cond].to_(device)

    def state_dict(self):
        conditions = sorted(self.running_topk.keys())
        result = dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            conditions=conditions)
        for i, c in enumerate(conditions):
            result.update({
                '%d.%s' % (i, k): v
                for k, v in self.running_topk[c].state_dict().items()})
        return result

    def set_state_dict(self, dic):
        conditions = list(dic['conditions'])
        subdicts = defaultdict(dict)
        for k, v in dic.items():
            if '.' in k:
                p, s = k.split('.', 1)
                subdicts[p][s] = v
        self.running_topk = {
            c: RunningTopK(state=subdicts[str(i)])
            for i, c in enumerate(conditions)}


class GatherTensor:
    """
    A tensor for gathering results, allocated and shaped on first insert.
    Creaed by tally.gather_topk for gathering topk visualizations.
    """

    def __init__(self, topk=None, data_shape=None, k=None, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        if k is None and topk is not None:
            k = topk.k
        if data_shape is None and topk is not None:
            data_shape = topk.data_shape
        assert k is not None
        assert data_shape is not None
        self.k = k
        self.data_shape = data_shape
        self._grid = None
        self._queue = defaultdict(list)

    def add(self, index, rank, data):
        if self._grid is None:
            # Allocation: pick up data shape from add.
            shape = self.data_shape
            if isinstance(shape, int):
                shape = (shape,)
            shape = shape + (self.k,) + data.shape
            self._grid = torch.zeros(shape, dtype=data.dtype)
        self._queue[index].append((rank, data))
        if len(self._queue) > len(self._grid) // 2:
            self._flush_queue()

    def _flush_queue(self):
        if len(self._queue):
            for index in sorted(self._queue.keys()):
                for rank, data in self._queue[index]:
                    self._grid[index][rank] = data
            self._queue.clear()

    def to_(self, device):
        self._flush_queue()
        if self._grid is not None:
            self._grid = self._grid.to(device)

    def state_dict(self):
        self._flush_queue()
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            k=self.k,
            data_shape=tuple(self.data_shape),
            grid=self._grid.cpu().numpy())

    def result(self):
        self._flush_queue()
        return self._grid

    def set_state_dict(self, dic):
        self.k = dic['k'].item()
        self.data_shape = tuple(dic['data_shape'])
        self._grid = torch.from_numpy(dic['grid'])
        self._queue = defaultdict(list)


class RunningQuantile:
    """
    Streaming randomized quantile computation for torch.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates (or old-style percentiles) can be read out using
    quantiles(q) or percentiles(p).

    Implemented as a sorted sample that retains at least r samples
    (by default r = 3072); the number of retained samples will grow to
    a finite ceiling as the data is accumulated.  Accuracy scales according
    to r: the default is to set resolution to be accurate to better than about
    0.1%, while limiting storage to about 50,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(self, r=3 * 1024, buffersize=None, seed=None,
                 state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.depth = None
        self.dtype = None
        self.device = None
        resolution = r * 2  # sample array is at least half full before discard
        self.resolution = resolution
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = None
        self.firstfree = [0]
        self.randbits = torch.ByteTensor(resolution)
        self.currentbit = len(self.randbits) - 1
        self.extremes = None
        self.count = 0
        self.batchcount = 0

    def size(self):
        return self.count

    def _lazy_init(self, incoming):
        self.depth = incoming.shape[1]
        self.dtype = incoming.dtype
        self.device = incoming.device
        self.data = [torch.zeros(self.depth, self.resolution,
                                 dtype=self.dtype, device=self.device)]
        self.extremes = torch.zeros(self.depth, 2,
                                    dtype=self.dtype, device=self.device)
        self.extremes[:, 0] = float('inf')
        self.extremes[:, -1] = -float('inf')

    def to_(self, device):
        """Switches internal storage to specified device."""
        if device != self.device:
            old_data = self.data
            old_extremes = self.extremes
            self.data = [d.to(device) for d in self.data]
            self.extremes = self.extremes.to(device)
            self.device = self.extremes.device
            del old_data
            del old_extremes

    def add(self, incoming):
        if self.depth is None:
            self._lazy_init(incoming)
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth, (incoming.shape[1], self.depth)
        self.count += incoming.shape[0]
        self.batchcount += 1
        # Convert to a flat torch array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = int(math.ceil(self.buffersize / self.samplerate))
        for index in range(0, len(incoming), chunksize):
            batch = incoming[index:index + chunksize]
            sample = sample_portion(batch, self.samplerate)
            if len(sample):
                self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        # First time sampling - the data source is very large.
                        self._scan_extremes(incoming)
                    incoming = sample_portion(incoming, self.samplerate)
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:, ff:ff + copycount] = torch.t(
                incoming[index:index + copycount, :])
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
                -(-self.data[index - 1].shape[1] // 2) if index else 1):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:, 0:self.firstfree[index]]
            data = data.sort()[0]
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:, 0], data[:, -1])
            offset = self._randbit()
            position = self.firstfree[index + 1]
            subset = data[:, offset::2]
            self.data[index + 1][:, position:position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1
        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
            torch.min(incoming, dim=0)[0],
            torch.max(incoming, dim=0)[0])

    def _update_extremes(self, minr, maxr):
        self.extremes[:, 0] = torch.min(
            torch.stack([self.extremes[:, 0], minr]), dim=0)[0]
        self.extremes[:, -1] = torch.max(
            torch.stack([self.extremes[:, -1], maxr]), dim=0)[0]

    def _randbit(self):
        self.currentbit += 1
        if self.currentbit >= len(self.randbits):
            self.randbits.random_(to=2)
            self.currentbit = 0
        return self.randbits[self.currentbit]

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            resolution=self.resolution,
            depth=self.depth,
            buffersize=self.buffersize,
            samplerate=self.samplerate,
            data=[d.cpu().detach().numpy()[:, :f].T
                  for d, f in zip(self.data, self.firstfree)],
            sizes=[d.shape[1] for d in self.data],
            extremes=self.extremes.cpu().detach().numpy(),
            size=self.count,
            batchcount=self.batchcount)

    def set_state_dict(self, dic):
        self.resolution = int(dic['resolution'])
        self.randbits = torch.ByteTensor(self.resolution)
        self.currentbit = len(self.randbits) - 1
        self.depth = int(dic['depth'])
        self.buffersize = int(dic['buffersize'])
        self.samplerate = float(dic['samplerate'])
        firstfree = []
        buffers = []
        for d, s in zip(dic['data'], dic['sizes']):
            firstfree.append(d.shape[0])
            buf = numpy.zeros((d.shape[1], s), dtype=d.dtype)
            buf[:, :d.shape[0]] = d.T
            buffers.append(torch.from_numpy(buf))
        self.firstfree = firstfree
        self.data = buffers
        self.extremes = torch.from_numpy((dic['extremes']))
        self.count = int(dic['size'])
        self.batchcount = int(dic.get('batchcount', 0))
        self.dtype = self.extremes.dtype
        self.device = self.extremes.device

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, :self.firstfree[0]].t())
        return self.extremes.clone()

    def median(self):
        return self.quantiles([0.5])[:, 0]

    def mean(self):
        return self.integrate(lambda x: x) / self.count

    def variance(self):
        mean = self.mean()[:, None]
        return self.integrate(lambda x: (x - mean).pow(2)) / (self.count - 1)

    def stdev(self):
        return self.variance().sqrt()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(0, torch.zeros(self.depth, cap,
                                            dtype=self.dtype, device=self.device))
            self.firstfree.insert(0, 0)
        else:
            # Unless we're so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index - 1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level's buffer size (rounding up)
            if self.data[index - 1].shape[1] - (amount + position) >= (
                    -(-self.data[index - 2].shape[1] // 2) if (index - 1) else 1):
                self.data[index - 1][:, position:position + amount] = (
                    self.data[index][:, :amount])
                self.firstfree[index - 1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:, :amount]
                data = data.sort()[0]
                if index == 1:
                    self._update_extremes(data[:, 0], data[:, -1])
                offset = self._randbit()
                scrunched = data[:, offset::2]
                self.data[index][:, :scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = int(math.ceil(self.resolution * (0.67 ** len(self.data))))
        if cap < 2:
            return 0
        # Round up to the nearest multiple of 8 for better GPU alignment.
        cap = -8 * (-cap // 8)
        return max(self.buffersize, cap)

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, :self.firstfree[0]].t())
        size = sum(self.firstfree)
        weights = torch.FloatTensor(size)  # Floating point
        summary = torch.zeros(self.depth, size,
                              dtype=self.dtype, device=self.device)
        index = 0
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:, index:index + ff] = self.data[level][:, :ff]
            weights[index:index + ff] = 2.0 ** level
            index += ff
        assert index == summary.shape[1]
        if sort:
            summary, order = torch.sort(summary, dim=-1)
            weights = weights[order.view(-1).cpu()].view(order.shape)
            summary = torch.cat(
                [self.extremes[:, :1], summary,
                 self.extremes[:, 1:]], dim=-1)
            weights = torch.cat(
                [torch.zeros(weights.shape[0], 1), weights,
                 torch.zeros(weights.shape[0], 1)], dim=-1)
        return (summary, weights)

    def quantiles(self, quantiles, old_style=False):
        if not hasattr(quantiles, 'cpu'):
            quantiles = torch.tensor(quantiles)
        qshape = quantiles.shape
        if self.count == 0:
            return torch.full((self.depth,) + qshape, torch.nan)
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        if old_style:
            # To be convenient with torch.percentile
            cumweights -= cumweights[:, 0:1].clone()
            cumweights /= cumweights[:, -1:].clone()
        else:
            cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros(self.depth, quantiles.numel(),
                             dtype=self.dtype, device=self.device)
        # numpy is needed for interpolation
        nq = quantiles.view(-1).cpu().detach().numpy()
        ncw = cumweights.cpu().detach().numpy()
        nsm = summary.cpu().detach().numpy()
        for d in range(self.depth):
            result[d] = torch.tensor(numpy.interp(nq, ncw[d], nsm[d]),
                                     dtype=self.dtype, device=self.device)
        return result.view((self.depth,) + qshape)

    def integrate(self, fun):
        result = None
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            term = torch.sum(
                fun(self.data[level][:, :ff]) * (2.0 ** level),
                dim=-1)
            if result is None:
                result = term
            else:
                result += term
        if result is not None:
            result /= self.samplerate
        return result

    def percentiles(self, percentiles):
        return self.quantiles(percentiles, old_style=True)

    def readout(self, count=1001, old_style=True):
        return self.quantiles(
            torch.linspace(0.0, 1.0, count), old_style=old_style)

    def normalize(self, data):
        '''
        Given input data as taken from the training distirbution,
        normalizes every channel to reflect quantile values,
        uniformly distributed, within [0, 1].
        '''
        assert self.count > 0
        assert data.shape[0] == self.depth
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros_like(data).float()
        # numpy is needed for interpolation
        ndata = data.cpu().numpy().reshape((data.shape[0], -1))
        ncw = cumweights.cpu().numpy()
        nsm = summary.cpu().numpy()
        for d in range(self.depth):
            normed = torch.tensor(numpy.interp(ndata[d], nsm[d], ncw[d]),
                                  dtype=torch.float, device=data.device).clamp_(0.0, 1.0)
            if len(data.shape) > 1:
                normed = normed.view(*(data.shape[1:]))
            result[d] = normed
        return result


class RunningConditionalQuantile:
    '''
    Equivalent to a map from conditions (any python hashable type)
    to RunningQuantiles.  The reason for the type is to allow limited
    GPU memory to be exploited while counting quantile stats on many
    different conditions, a few of which are common and which benefit
    from GPU, but most of which are rare and would not all fit into
    GPU RAM.

    To move a set of conditions to a device, use rcq.to_(device, conds).
    Then in the future, move the tallied data to the device before
    calling rcq.add, that is, rcq.add(cond, data.to(device)).

    To allow the caller to decide which conditions to allow to use GPU,
    rcq.most_common_conditions(n) returns a list of the n most commonly
    added conditions so far.
    '''

    def __init__(self, r=3 * 1024, buffersize=None, seed=None,
                 state=None):
        self.first_rq = None
        self.call_stats = defaultdict(int)
        self.running_quantiles = {}
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.rq_args = dict(r=r, buffersize=buffersize,
                            seed=seed)

    def add(self, condition, incoming):
        if condition not in self.running_quantiles:
            self.running_quantiles[condition] = RunningQuantile(**self.rq_args)
            if self.first_rq is None:
                self.first_rq = self.running_quantiles[condition]
        self.call_stats[condition] += 1
        rq = self.running_quantiles[condition]
        # For performance reasons, the caller can move some conditions to
        # the CPU if they are not among the most common conditions.
        if rq.device is not None and (rq.device != incoming.device):
            rq.to_(incoming.device)
        self.running_quantiles[condition].add(incoming)

    def most_common_conditions(self, n):
        return sorted(self.call_stats.keys(),
                      key=lambda c: -self.call_stats[c])[:n]

    def collected_add(self, conditions, incoming):
        for c in conditions:
            self.add(c, incoming)

    def keys(self):
        return self.running_quantiles.keys()

    def sizes(self):
        return {k: self.running_quantiles[k].size() for k in self.keys()}

    def conditional(self, c):
        return self.running_quantiles[c]

    def has_conditional(self, c):
        return c in self.running_quantiles

    def collected_quantiles(self, conditions, quantiles, old_style=False):
        result = torch.zeros(
            size=(len(conditions), self.first_rq.depth, len(quantiles)),
            dtype=self.first_rq.dtype,
            device=self.first_rq.device)
        for i, c in enumerate(conditions):
            if c in self.running_quantiles:
                result[i] = self.running_quantiles[c].quantiles(
                    quantiles, old_style)
        return result

    def collected_normalize(self, conditions, values):
        result = torch.zeros(
            size=(len(conditions), values.shape[0], values.shape[1]),
            dtype=torch.float,
            device=self.first_rq.device)
        for i, c in enumerate(conditions):
            if c in self.running_quantiles:
                result[i] = self.running_quantiles[c].normalize(values)
        return result

    def to_(self, device, conditions=None):
        if conditions is None:
            conditions = self.keys()
        for cond in conditions:
            if cond in self.running_quantiles:
                self.running_quantiles[cond].to_(device)

    def state_dict(self):
        conditions = sorted(self.running_quantiles.keys())
        result = dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            rq_args=self.rq_args,
            conditions=conditions)
        for i, c in enumerate(conditions):
            result.update({
                '%d.%s' % (i, k): v
                for k, v in self.running_quantiles[c].state_dict().items()})
        return result

    def set_state_dict(self, dic):
        self.rq_args = dic['rq_args'].item()
        conditions = list(dic['conditions'])
        subdicts = defaultdict(dict)
        for k, v in dic.items():
            if '.' in k:
                p, s = k.split('.', 1)
                subdicts[p][s] = v
        self.running_quantiles = {
            c: RunningQuantile(state=subdicts[str(i)])
            for i, c in enumerate(conditions)}
        if conditions:
            self.first_rq = self.running_quantiles[conditions[0]]

    # example usage:
    # levels = rqc.conditional(()).quantiles(1 - fracs)
    # denoms = 1 - rqc.collected_normalize(cats, levels)
    # isects = 1 - rqc.collected_normalize(labels, levels)
    # unions = fracs + denoms[cats] - isects
    # iou = isects / unions


class RunningVariance:
    '''
    Running computation of mean and variance. Use this when you just need
    basic stats without covariance.
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self.batchcount = 0
        self._mean = None
        self.v_cmom2 = None

    def add(self, a):
        if len(a.shape) == 1:
            a = a[None, :]
        if len(a.shape) > 2:
            a = (a.view(a.shape[0], a.shape[1], -1).permute(0, 2, 1)
                 .reshape(-1, a.shape[1]))
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        centered = a - batch_mean
        self.batchcount += 1
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            self.v_cmom2 = centered.pow(2).sum(0)
            return
        # Update a batch using Chan-style update for numerical stability.
        oldcount = self.count
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)
        # Update the variance using the batch deviation
        self.v_cmom2.add_(centered.pow(2).sum(0))
        self.v_cmom2.add_(delta.pow_(2).mul_(new_frac * oldcount))

    def size(self):
        return self.count

    def mean(self):
        return self._mean

    def variance(self):
        return self.v_cmom2 / (self.count - 1)

    def stdev(self):
        return self.variance().sqrt()

    def to_(self, device):
        self._mean = self._mean.to(device)
        self.v_cmom2 = self.v_cmom2.to(device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            count=self.count,
            batchcount=self.batchcount,
            mean=self._mean.cpu().numpy(),
            cmom2=self.v_cmom2.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self.batchcount = dic['batchcount'].item()
        self._mean = torch.from_numpy(dic['mean'])
        self.v_cmom2 = torch.from_numpy(dic['cmom2'])


class RunningConditionalVariance:
    def __init__(self, state=None):
        self.running_var = {}
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return

    def add(self, condition, incoming):
        if condition not in self.running_var:
            self.running_var[condition] = RunningVariance()
        rv = self.running_var[condition]
        rv.add(incoming)

    def collected_add(self, conditions, incoming):
        for c in conditions:
            self.add(c, incoming)

    def keys(self):
        return self.running_var.keys()

    def conditional(self, c):
        return self.running_var[c]

    def has_conditional(self, c):
        return c in self.running_var

    def to_(self, device, conditions=None):
        if conditions is None:
            conditions = self.keys()
        for cond in conditions:
            if cond in self.running_var:
                self.running_var[cond].to_(device)

    def state_dict(self):
        conditions = sorted(self.running_var.keys())
        result = dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            conditions=conditions)
        for i, c in enumerate(conditions):
            result.update({
                '%d.%s' % (i, k): v
                for k, v in self.running_var[c].state_dict().items()})
        return result

    def set_state_dict(self, dic):
        conditions = list(dic['conditions'])
        subdicts = defaultdict(dict)
        for k, v in dic.items():
            if '.' in k:
                p, s = k.split('.', 1)
                subdicts[p][s] = v
        self.running_var = {
            c: RunningVariance(state=subdicts[str(i)])
            for i, c in enumerate(conditions)}


class RunningCrossCovariance:
    '''
    Running computation. Use this when an off-diagonal block of the
    covariance matrix is needed (e.g., when the whole covariance matrix
    does not fit in the GPU).

    Chan-style numerically stable update of mean and full covariance matrix.
    Chan, Golub. LeVeque. 1983. http://www.jstor.org/stable/2683386
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self._mean = None
        self.cmom2 = None
        self.v_cmom2 = None

    def add(self, a, b):
        if len(a.shape) == 1:
            a = a[None, :]
            b = b[None, :]
        assert(a.shape[0] == b.shape[0])
        if len(a.shape) > 2:
            a, b = [d.view(d.shape[0], d.shape[1], -1).permute(0, 2, 1)
                    .reshape(-1, d.shape[1]) for d in [a, b]]
        batch_count = a.shape[0]
        batch_mean = [d.sum(0) / batch_count for d in [a, b]]
        centered = [d - bm for d, bm in zip([a, b], batch_mean)]
        # If more than 10 billion operations, divide into batches.
        sub_batch = -(-(10 << 30) // (a.shape[1] * b.shape[1]))
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            self.v_cmom2 = [c.pow(2).sum(0) for c in centered]
            self.cmom2 = a.new(a.shape[1], b.shape[1]).zero_()
            progress_addbmm(self.cmom2, centered[0][:, :, None],
                            centered[1][:, None, :], sub_batch)
            return
        # Update a batch using Chan-style update for numerical stability.
        oldcount = self.count
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = [bm.sub_(m).mul_(new_frac)
                 for bm, m in zip(batch_mean, self._mean)]
        for m, d in zip(self._mean, delta):
            m.add_(d)
        # Update the cross-covariance using the batch deviation
        progress_addbmm(self.cmom2, centered[0][:, :, None],
                        centered[1][:, None, :], sub_batch)
        self.cmom2.addmm_(alpha=new_frac * oldcount,
                          mat1=delta[0][:, None], mat2=delta[1][None, :])
        # Update the variance using the batch deviation
        for c, vc2, d in zip(centered, self.v_cmom2, delta):
            vc2.add_(c.pow(2).sum(0))
            vc2.add_(d.pow_(2).mul_(new_frac * oldcount))

    def mean(self):
        return self._mean

    def variance(self):
        return [vc2 / (self.count - 1) for vc2 in self.v_cmom2]

    def stdev(self):
        return [v.sqrt() for v in self.variance()]

    def covariance(self):
        return self.cmom2 / (self.count - 1)

    def correlation(self):
        covariance = self.covariance()
        rstdev = [s.reciprocal() for s in self.stdev()]
        cor = rstdev[0][:, None] * covariance * rstdev[1][None, :]
        # Remove NaNs
        cor[torch.isnan(cor)] = 0
        return cor

    def to_(self, device):
        self._mean = [m.to(device) for m in self._mean]
        self.v_cmom2 = [vcs.to(device) for vcs in self.v_cmom2]
        self.cmom2 = self.cmom2.to(device)

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            count=self.count,
            mean_a=self._mean[0].cpu().numpy(),
            mean_b=self._mean[1].cpu().numpy(),
            cmom2_a=self.v_cmom2[0].cpu().numpy(),
            cmom2_b=self.v_cmom2[1].cpu().numpy(),
            cmom2=self.cmom2.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self._mean = [torch.from_numpy(dic[k]) for k in ['mean_a', 'mean_b']]
        self.v_cmom2 = [torch.from_numpy(dic[k])
                        for k in ['cmom2_a', 'cmom2_b']]
        self.cmom2 = torch.from_numpy(dic['cmom2'])


class RunningCovariance:
    '''
    Running computation. Use this when the entire covariance matrix is needed,
    and when the whole covariance matrix fits in the GPU.

    Chan-style numerically stable update of mean and full covariance matrix.
    Chan, Golub. LeVeque. 1983. http://www.jstor.org/stable/2683386
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self._mean = None
        self.cmom2 = None

    def add(self, a):
        if len(a.shape) == 1:
            a = a[None, :]
        batch_count = a.shape[0]
        batch_mean = a.sum(0) / batch_count
        centered = a - batch_mean
        # If more than 10 billion operations, divide into batches.
        sub_batch = -(-(10 << 30) // (a.shape[1] * a.shape[1]))
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            self.cmom2 = a.new(a.shape[1], a.shape[1]).zero_()
            progress_addbmm(self.cmom2, centered[:, :, None], centered[:, None, :],
                            sub_batch)
            return
        # Update a batch using Chan-style update for numerical stability.
        oldcount = self.count
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = batch_mean.sub_(self._mean).mul_(new_frac)
        self._mean.add_(delta)
        # Update the variance using the batch deviation
        progress_addbmm(self.cmom2, centered[:, :, None], centered[:, None, :],
                        sub_batch)
        self.cmom2.addmm_(
            alpha=new_frac * oldcount, mat1=delta[:, None], mat2=delta[None, :])

    def cpu_(self):
        self._mean = self._mean.cpu()
        self.cmom2 = self.cmom2.cpu()

    def cuda_(self):
        self._mean = self._mean.cuda()
        self.cmom2 = self.cmom2.cuda()

    def to_(self, device):
        self._mean, self.cmom2 = [m.to(device)
                                  for m in [self._mean, self.cmom2]]

    def mean(self):
        return self._mean

    def covariance(self):
        return self.cmom2 / self.count

    def covariancePSD(self):
        return nearestCov(self.covariance())

    def correlation(self):
        covariance = self.covariance()
        rstdev = covariance.diag().sqrt().reciprocal()
        return rstdev[:, None] * covariance * rstdev[None, :]

    def correlationPSD(self):
        return nearestCorr(self.correlation())

    def variance(self):
        return self.covariance().diag()

    def stdev(self):
        return self.variance().sqrt()

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            count=self.count,
            mean=self._mean.cpu().numpy(),
            cmom2=self.cmom2.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self._mean = torch.from_numpy(dic['mean'])
        self.cmom2 = torch.from_numpy(dic['cmom2'])


class RunningSecondMoment:
    '''
    Running computation. Use this when the entire non-centered 2nd-moment
    "covariance-like" matrix is needed, and when the whole matrix fits
    in the GPU.
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self.mom2 = None

    def add(self, a):
        if len(a.shape) == 1:
            a = a[None, :]
        # Initial batch reveals the shape of the data.
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        batch_count = a.shape[0]
        # If more than 10 billion operations, divide into batches.
        sub_batch = -(-(10 << 30) // (a.shape[1] * a.shape[1]))
        # Update the covariance using the batch deviation
        self.count += batch_count
        progress_addbmm(self.mom2, a[:, :, None], a[:, None, :], sub_batch)

    def cpu_(self):
        self.mom2 = self.mom2.cpu()

    def cuda_(self):
        self.mom2 = self.mom2.cuda()

    def to_(self, device):
        self.mom2 = self.mom2.to(device)

    def moment(self):
        return self.mom2 / self.count

    def momentPSD(self):
        return nearestCov(self.moment())

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            count=self.count,
            mom2=self.mom2.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self.mom2 = torch.from_numpy(dic['mom2'])


class RunningBincount:
    '''
    Running bincount.  The counted array should be an integer type with
    non-negative integers.  Also
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self._bincount = None

    def add(self, a, size=None):
        a = a.view(-1)
        bincount = a.bincount()
        if self._bincount is None:
            self._bincount = bincount
        elif len(self._bincount) < len(bincount):
            bincount[:len(self._bincount)] += self._bincount
            self._bincount = bincount
        else:
            self._bincount[:len(bincount)] += bincount
        if size is None:
            self.count += len(a)
        else:
            self.count += size

    def cpu_(self):
        self._bincount = self._bincount.cpu()

    def cuda_(self):
        self._bincount = self._bincount.cuda()

    def to_(self, device):
        self._bincount = self._bincount.to(device)

    def size(self):
        return self.count

    def mean(self):
        return (self._bincount).float() / self.count

    def bincount(self):
        return self._bincount

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            count=self.count,
            bincount=self._bincount.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self._bincount = torch.from_numpy(dic['bincount'])


def progress_addbmm(accum, x, y, batch_size):
    '''
    Break up very large adbmm operations into batches so progress can be seen.
    '''
    from . import pbar
    if x.shape[0] <= batch_size:
        return accum.addbmm_(x, y)
    for i in pbar(range(0, x.shape[0], batch_size), desc='bmm'):
        accum.addbmm_(x[i:i + batch_size], y[i:i + batch_size])
    return accum


def sample_portion(vec, p=0.5):
    bits = torch.bernoulli(torch.zeros(vec.shape[0], dtype=torch.uint8,
                                       device=vec.device), p)
    return vec[bits]


def resolve_state_dict(s):
    if isinstance(s, str):
        return numpy.load(s, allow_pickle=True)
    return s


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("error")
    import time
    import argparse
    parser = argparse.ArgumentParser(
        description='Test things out')
    parser.add_argument('--mode', default='cpu', help='cpu or cuda')
    parser.add_argument('--test_size', type=int, default=1000000)
    args = parser.parse_args()

    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    amount = args.test_size
    quantiles = 1000
    data = numpy.arange(float(amount))
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    depth = 50
    test_cuda = torch.cuda.is_available()
    alldata = data[:, None] + (numpy.arange(depth) * amount)[None, :]
    actual_sum = torch.FloatTensor(numpy.sum(alldata * alldata, axis=0))
    amt = amount // depth
    for r in range(depth):
        numpy.random.shuffle(alldata[r * amt:r * amt + amt, r])
    if args.mode == 'cuda':
        alldata = torch.cuda.FloatTensor(alldata)
        dtype = torch.float
        device = torch.device('cuda')
    else:
        alldata = torch.FloatTensor(alldata)
        dtype = torch.float
        device = None
    starttime = time.time()
    qc = RunningQuantile(r=3 * 1024)
    qc.add(alldata)
    # Test state dict
    saved = qc.state_dict()
    # numpy.savez('foo.npz', **saved)
    # saved = numpy.load('foo.npz')
    qc = RunningQuantile(state=saved)
    assert not qc.device.type == 'cuda'
    qc.add(alldata)
    actual_sum *= 2
    ro = qc.readout(1001).cpu()
    endtime = time.time()
    gt = torch.linspace(0, amount, quantiles + 1)[None, :] + (
        torch.arange(qc.depth, dtype=torch.float) * amount)[:, None]
    maxreldev = torch.max(torch.abs(ro - gt) / amount) * quantiles
    print("Maximum relative deviation among %d perentiles: %f" % (
        quantiles, maxreldev))
    minerr = torch.max(torch.abs(qc.minmax().cpu()[:, 0] -
                                 torch.arange(qc.depth, dtype=torch.float) * amount))
    maxerr = torch.max(torch.abs((qc.minmax().cpu()[:, -1] + 1) -
                                 (torch.arange(qc.depth, dtype=torch.float) + 1) * amount))
    print("Minmax error %f, %f" % (minerr, maxerr))
    interr = torch.max(torch.abs(qc.integrate(lambda x: x * x).cpu()
                                 - actual_sum) / actual_sum)
    print("Integral error: %f" % interr)
    medianerr = torch.max(torch.abs(qc.median() -
                                    alldata.median(0)[0]) / alldata.median(0)[0]).cpu()
    print("Median error: %f" % interr)
    meanerr = torch.max(
        torch.abs(qc.mean() - alldata.mean(0)) / alldata.mean(0)).cpu()
    print("Mean error: %f" % meanerr)
    varerr = torch.max(
        torch.abs(qc.variance() - alldata.var(0)) / alldata.var(0)).cpu()
    print("Variance error: %f" % varerr)
    counterr = ((qc.integrate(lambda x: torch.ones(x.shape[-1]).cpu())
                 - qc.size()) / (0.0 + qc.size())).item()
    print("Count error: %f" % counterr)
    print("Time %f" % (endtime - starttime))
    # Algorithm is randomized, so some of these will fail with low probability.
    assert maxreldev < 1.0
    assert minerr == 0.0
    assert maxerr == 0.0
    assert interr < 0.01
    assert abs(counterr) < 0.001
    print("OK")


class RunningAllIntersectionAndUnion:
    '''
    Running computation of intersections and unions of two binary vectors.
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self.intersection = None
        self.total_a = None
        self.total_b = None

    def add(self, S, G):
        assert len(S.shape) == 2 and len(G.shape) == 2
        assert S.dtype == torch.bool and G.dtype == torch.bool
        assert len(S) == len(G), f'{len(S)} vs {len(G)}'
        S = S.float()  # CUDA only supports mm on float...
        G = G.float()  # otherwise we would use integers.
        intersection = torch.mm(S.t(), G)
        ssum = S.sum(0)
        gsum = G.sum(0)
        if self.intersection is None:
            self.intersection = intersection
            self.total_a = ssum
            self.total_b = gsum
        else:
            self.intersection += intersection
            self.total_a += ssum
            self.total_b += gsum
        self.count += len(S)

    def size(self):
        return self.count

    def iou(self):
        union = self.total_a[:, None] + self.total_b[None, :] - self.intersection
        out = self.intersection / (union + 1e-20)
        return out

    def to_(self, _device):
        self.total_a = self.total_a.to(_device)
        self.total_b = self.total_b.to(_device)
        self.intersection = self.intersection.to(_device)

    def state_dict(self):
        return dict(constructor=self.__module__ + '.' +
                    self.__class__.__name__ + '()',
                    count=self.count,
                    total_a=self.total_a.cpu().numpy(),
                    total_b=self.total_b.cpu().numpy(),
                    intersection=self.intersection.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self.total_a = torch.tensor(dic['total_a'])
        self.total_b = torch.tensor(dic['total_b'])
        self.intersection = torch.tensor(dic['intersection'])


class RunningConditionalVariance:
    def __init__(self, state=None):
        self.running_var = {}
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return

    def add(self, condition, incoming):
        if condition not in self.running_var:
            self.running_var[condition] = RunningVariance()
        rv = self.running_var[condition]
        rv.add(incoming)

    def collected_add(self, conditions, incoming):
        for c in conditions:
            self.add(c, incoming)

    def keys(self):
        return self.running_var.keys()

    def conditional(self, c):
        return self.running_var[c]

    def has_conditional(self, c):
        return c in self.running_var

    def to_(self, device, conditions=None):
        if conditions is None:
            conditions = self.keys()
        for cond in conditions:
            if cond in self.running_var:
                self.running_var[cond].to_(device)

    def state_dict(self):
        conditions = sorted(self.running_var.keys())
        result = dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            conditions=conditions)
        for i, c in enumerate(conditions):
            result.update({
                '%d.%s' % (i, k): v
                for k, v in self.running_var[c].state_dict().items()})
        return result

    def set_state_dict(self, dic):
        conditions = list(dic['conditions'])
        subdicts = defaultdict(dict)
        for k, v in dic.items():
            if '.' in k:
                p, s = k.split('.', 1)
                subdicts[p][s] = v
        self.running_var = {
            c: RunningVariance(state=subdicts[str(i)])
            for i, c in enumerate(conditions)}


from statsmodels.stats.correlation_tools import cov_nearest, corr_nearest
def nearestCov(A):
    '''
    Find the positive-semidefinite near to A with the corlelation
    matrix nearest A.
    '''
    npA = A.detach().cpu().double().numpy()
    npPD = cov_nearest(A, method='nearest')
    return torch.from_numpy(npPD).to(A.device, A.dtype)

def nearestCorr(A):
    '''
    Find the positive-semidefinite near to A with the corlelation
    matrix nearest A.
    '''
    npA = A.detach().cpu().double().numpy()
    npPD = corr_nearest(A)
    return torch.from_numpy(npPD).to(A.device, A.dtype)


from numpy import linalg
import numpy
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A pytorch port of Ahmed Fasih's Numpy port [1] of John D'Errico's
    `nearestSPD` MATLAB code [2], which credits [3].

    [1] https://stackoverflow.com/a/43244194/265298

    [2] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [3] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = torch.svd(B)
    H = torch.mm(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = numpy.spacing(linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        B.choleksy()
        return True
    except la.LinAlgError:
        return False

