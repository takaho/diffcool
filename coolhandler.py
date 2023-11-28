import subprocess
import scipy.io, scipy.sparse
import argparse, sys, re, os
import h5py
import numpy as np
import pandas as pd
import logging
import json
import skimage, skimage.io
import numba
import collections
import skimage
import skimage.io
import logging

sys.path.append('/mnt/nas/genomeplatform/scripts')
import tkutil    

class HiCMatrix(object):
    """Class for Hi-C matrix
    """
    def __init__(self, filename, chrom_row, chrom_col, count_matrix, binsize, weight_row=None, weight_col=None):
        """Initialization of Hi-C instance
        Args:
            filename (str): filename (cool/mcool)
            chrom_row (int): _description_
            chrom_col (int): _description_
            count_matrix (_type_): _description_
            binsize (int): _description_
            weight_row (ndarray, optional): _description_. Defaults to None.
            weight_col (ndarray, optional): _description_. Defaults to None.
        """
        self.__filename = filename
        self.__chromosomes = [chrom_row, chrom_col]
        self.__data = count_matrix
        self.__binsize = binsize
        self.__weights = None #if weight_row is None or weight_col is None else [weight_row, weight_col]
        self.__balanced = None
        if weight_row is not None and weight_col is not None:
            self.set_weights(weight_row, weight_col)
        
    def set_weights(self, weight_row, weight_col):
        if weight_row.size == self.__data.shape[0] and weight_col.size == self.__data.shape[1]:
            weight_row[np.isnan(weight_row)] = 0
            weight_col[np.isnan(weight_col)] = 0
            self.__weights = [weight_row, weight_col]
            self.__balanced = None
        else:
            sys.stderr.write('failed to set weight {}x{} / w {}, {}'.format(self.__data.shape[0], self.__data.shape[1], weight_row.size, weight_col.size))
            raise Exception('failed to set weight {}x{} / w {}, {}'.format(self.__data.shape[0], self.__data.shape[1], weight_row.size, weight_col.size))
            
    def transpose(self):
        self.__chromosomes = [self.__chromosomes[1], self.chromosomes[0]]        
        if self.__weights is not None:
            self.__weights = [self.__weights[1], self.__weights[0]]
        self.__data = self.__data.T
        if self.__balanced:
            self.__balanced = self.__balanced.T
            
    def __balance(self):
        if self.__balanced is not None or self.__weights is None:
            # print('alread balanced')
            return None
        T = self.__weights[0].reshape(-1, 1) * self.__weights[1].reshape(1, -1)
        if self.__weights is not None and self.__data is not None and self.__weights[0].size == self.__data.shape[0] and self.__weights[1].size == self.__data.shape[1]:
            self.__balanced = (self.__data.toarray() * T).astype(np.float32)#(self.__weights[0].reshape(-1, 1) * self.__weights[1].reshape(1, -1))).astype(np.float32)
        else:
            self.__balanced = None
        return self.__balanced
    
    def __get_data(self):
        if self.__balance() is None:
            return self.__data
        return self.__balanced
    
    filename = property(lambda s:s.__filename)
    chromosomes = property(lambda s:s.__chromosomes)
    is_balanced = property(lambda s:s.__is_balanced is not None)
    binsize = property(lambda s:s.__binsize)
    shape = property(lambda s:s.__data.shape)
    data = property(__get_data)
    rawcount = property(lambda s:s.__data)
    shape = property(lambda s:s.__data.shape)

class HiCSubregion(object):
    """HiC regional data for local structure

    Args:
        HiCSubregion(filename, chromosome, start, stop)
    """
    def __init__(self, filename, chromosome, start, stop, data=None, index_start=-1, index_end=-1, binsize=-1):
        self.__filename = filename
        self.__title = os.path.splitext(os.path.basename(filename))[0]
        self.__position = [chromosome, start, stop]
        self.__binsize = 0
        self.__matrix = None
        self.__binsize = 1
        self.__total = 0
        self.__index_range = index_start, index_end
        self.__weight = None
        self.__normalized = None
        self.__balanced = None
        if data is not None:
            self.set_matrix(data, binsize)
        pass

    def __normalize(self):
        if self.__balanced is not None:
            print('alread balanced')
            return
        if self.__weight is not None and self.__matrix is not None and self.__weight.size == self.__matrix.shape[0] == self.__matrix.shape[1]:
            self.__balanced = self.__matrix * (self.__weight.reshape(-1, 1) * self.__weight.reshape(1, -1))
        else:
            self.__balanced = None

    def set_weight(self, weight, index_start=-1, index_end=-1):
        if index_start >= 0 and index_start < index_end:
            self.__weight = np.array(weight[index_start:index_end], np.float32)
            self.__index_range = index_start, index_end
        else:
            self.__weight = np.array(weight, np.float32)
        self.__normalize()

    def set_matrix(self, matrix, binsize=-1):
        if binsize < 0:
            n = matrix.shape[0]
            binsize = max(int(np.ceil((self.end - self.start) / n)), 1)
        self.__binsize = binsize
        self.__matrix = matrix
        self.__total = np.sum(matrix)
        self.__normalize()
    def __repr__(self):
        return '{}\t{}:{}-{}\tbinsize={}, total={}'.format(
            self.title, self.chromosome, self.start, self.end, 
            self.binsize, self.total)
    def dump(self, filename):
        dirname = os.path.dirname(os.path.abspath(filename))
        # print(dirname)
        if os.path.exists(dirname) is False:
            os.makedirs(dirname, exist_ok=True)
        fields = {
            'filename':self.filename,
            'title':self.title,
            'position':[self.chromosome, self.start, self.end],
            'binsize':self.binsize,
            'total':self.total,
            'index_range':self.index_range,
            'weight':self.weight,
            'normalized':self.__normalized,
            'balanced':self.__balanced
        }
        if self.__matrix is not None:
            fn_mat = os.path.join(dirname, '.' + os.path.basename(filename) + '.mtx')
            scipy.io.mmwrite(fn_mat, scipy.sparse.coo_matrix(self.__matrix))
            try:
                fn_mat_gz = fn_mat + '.gz'
                if os.path.exists(fn_mat_gz):
                    os.unlink(fn_mat_gz)
                cmd = 'pigz', '-p', '4', fn_mat
                subprocess.Popen(cmd).wait()
                fn_mat = fn_mat_gz
            except:
                pass
            fields['matrix'] = fn_mat
        with open(filename, 'w') as fo:
            json.dump(fields, fo)
        pass
    @classmethod
    def load(filename):
        with open(filename) as fi:
            fields = json.load(fi)
        fn_mat = fields.get('matrix', None)
        if fn_mat:
            mat = scipy.io.mmread(fn_mat)
        else:
            mat = None
        pos = fields['position']
        ir = fields['index_range']
    
        obj = HiCSubregion(fields['filename'], pos[0], pos[1], pos[2], data=mat, index_start=ir[0], index_end=ir[1], binsize=fields['binsize'])
        obj.__weight = fields['weight']
        obj.__total = fields['total']
        obj.__normalized = fields['normalized']
        obj.__balanced = fields['balanced']
        return obj        

    title = property(lambda _:_.__title)
    filename = property(lambda _:_.__filename)
    chromosome = property(lambda _:_.__position[0])
    start = property(lambda _:int(_.__position[1]))
    end = property(lambda _:int(_.__position[2]))
    binsize = property(lambda _:int(_.__binsize))
    data = property(lambda _:_.__balanced if _.__balanced is not None else _.__matrix)
    normalized = property(lambda _:__balanced)
    dtype = property(lambda _:_.__matrix.dtype)
    total = property(lambda _:int(_.__total))
    index_range = property(lambda _:[int(x) for x in _.__index_range])
    weight = property(lambda _:[float(x) for x in _.__weight] if _.__weight is not None else None)
    has_weight = property(lambda _:_.__weight is not None)


@numba.njit('void(i4[:,:], f4, u1[:,:,:])')
def __convert_image_i4(data, threshold, img):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i,j]
            if val == 0:
                img[i,j,:] = 255
            else:
                d = min(1, val / threshold)
                r = 255 - np.uint8(d * 32)
                g = 255 - np.uint8(d * 192)
                b = 255 - np.uint8(d * 255)
                img[i,j] = (r, g, b)

@numba.njit('void(f4[:,:], f4, u1[:,:,:])')
def __convert_image_f4(data, threshold, img):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i,j]
            if val == 0:
                img[i,j,:] = 255
            else:
                d = min(1, val / threshold)
                r = 255 - np.uint8(d * 32)
                g = 255 - np.uint8(d * 192)
                b = 255 - np.uint8(d * 255)
                img[i,j] = (r, g, b)
    

def plot_matrix(matrix: HiCMatrix, **kwargs)->np.array:
    """Convert matrix to PNG image

    Args:
        matrix (HiCMatrix): HiCMatrix object
        converter : converting function
        percentile : (default 99.9)
        threshold : threshold value , mutual with percentile

    Raises:
        Exception: inconsistent shape

    Returns:
        np.array: PNG image
    """
    data = matrix.data
    converter = kwargs.get('converter', None)
    img = np.full((data.shape[0], data.shape[1], 3), (255,255,255), dtype=np.uint8)
    percentile = kwargs.get('percentile', 99.9)
    threshold = kwargs.get('threshold', 0)
    logger = kwargs.get('logger', logging.getLogger())
    rawmode = kwargs.get('rawmode', False)
    
    if rawmode:
        data = matrix.rawcount
    
    def __get_percentile(data, percentile):
        # print(data.shape, data.size, np.mean(data), percentile)
        if scipy.sparse.issparse(data):
            n = data.shape[0] * data.shape[1]
            values = data.data
            n_zero = n - values.size
            pt = max(0, percentile - n_zero * 100 / n)
            # print(n, data.size, n_zero, n_zero / n, percentile, pt, percentile / 100 - n_zero / n)
            threshold = np.percentile(values, pt)
        else:
            threshold = np.percentile(data, float(percentile))
        return np.float32(threshold)
        
    if threshold <= 0:
        threshold = __get_percentile(data, percentile)
    if scipy.sparse.issparse(data):
        # print('check sparse matrix dtype')
        if data.dtype in (np.int32, np.int64, np.uint32, np.uint64):
            dtype = np.int32
        else:
            dtype = np.float32
        # dtype = np.int32 if data.dtype in [np.issubclass(data.dtype, np.signedinteger) else np.float32
        # print(dtype, data.dtype, isinstance(data.dtype, np.signedinteger))
        pixels = data.astype(dtype).toarray()
        
    else:
        pixels = data
    # print(pixels.dtype, data.dtype)
    # exit()
    if converter is None:
        if data.dtype == np.int32:
            __convert_image_i4(pixels, threshold, img)
        elif data.dtype == np.float32:
            __convert_image_f4(pixels, np.float32(threshold), img)
        else:
            raise Exception('invalid type {} for matrix'.format(data.dtype))
    else:
        pixels = data.toarray()
        for i, j in [(i,j) for i in range(data.shape[0]) for j in range(data.shape[1])]:
            img[i,j] = converter(pixels[i,j])
    return img
    
# def detect_binsize(mcool):
def read_resolutions(filename:str)->np.array:
    """Read the resolutions from mcool file
    """
    if filename.endswith('.mcool'):
        with h5py.File(filename) as f5:
            resolutions = []
            for resol in f5['resolutions'].keys():
                if resol.isdigit():
                    resolutions.append(int(resol))
            f5.close()
        return np.array(resolutions, dtype=np.int32)
    else:
        clen = read_chromosomes(filename)
        l0 = list(clen.values())[0]
        # print('clen = ', clen)
        # print('length = ', l0)
        with h5py.File(filename) as f5:
            # print(f5['bins'])
            # print(f5['bins/chrom'])
            cidx = np.array(f5['bins/chrom'])
            binsize = 1
            for i, cid in enumerate(cidx):
                if i != 0:
                    binsize = l0 / i
                    break
            fig = int(np.log10(binsize) - 1)
            dist = []
            for f in range(fig-1, fig+2):
                for d in (1, 2, 2.5, 4, 5, 7.5, 8):
                    b = int(10 ** f * d)
                    dist.append((b, abs(b-binsize)))
            # print(dist)
            # print(list(sorted(dist, key=lambda x_:x_[1]))[0])
            binsize = list(sorted(dist, key=lambda x_:x_[1]))[0][0]
        # print(binsize)
        return [binsize, ]
    
def determine_filetype(filename:str)->str:
    """Cool or Mcool or not

    Args:
        filename (str): filename

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        str: 'cool' or 'mcool'
    """
    if filename.endswith('.mcool'):
        return 'mcool'
    elif filename.endswith('.cool'):
        return 'cool'
    try:
        with h5py.File(filename, 'r') as f5:
            try:
                r = f5['resolutions']
                if r is not None:
                    return 'mcool'
                r = f5['chroms/name']
                if r is not None:
                    return 'cool'
                f5.close()
            except:
                raise Exception(f'{filename} is not mcool nor cool')
    except Exception as e:
        raise
#
def __load_weight(f5:h5py.File, node:str):
    """ load weights from h5py.File
    """
    weight = list(f5[f'{node}/bins/weight'])
    return weight

# def list_chromosomes(filename:str, **kwargs)->dict:

def retrieve_chromosome_matrix(filename:str, chromosome:str, **kwargs)->HiCSubregion:
    """ retrieve the chromosome matrix
    """
    binsize = kwargs.get('resolution', 0)
    # print('binsize', binsize)
    chroms = read_chromosomes(filename)
    chr_idx = -1
    chr_length = 0
    resolutions = read_resolutions(filename)
    logger = kwargs.get('logger', logging.getLogger())
    # print(resolutions)
    if binsize not in resolutions:
        binsize = resolutions[0]
    # print('binsize', binsize)
    for i, c in enumerate(chroms.keys()):
        if chromosome == c:
            chr_idx = i
            chr_length = chroms[c]
            break
    logger.info('resolution={}, chromosome={} (index={}, {})'.format(binsize, chromosome, chr_idx, chr_length))
    # print('binsize', binsize)
    if chr_idx < 0: 
        raise Exception(f'no chromosome {chromosome} in {filename}')
        return None
    ft = determine_filetype(filename)
    if ft == 'mcool':
        base = f'resolutions/{binsize}/'
        pass
    elif ft == 'cool':
        base = ''
        pass
    logger.info(f'loading {filename} at "{base}"')
    with h5py.File(filename, 'r') as f5:
        cidx = np.array(f5[f'{base}bins/chrom'])
        left = cidx.size
        right = 0
        rng = [left, right]
        for i, c_ in enumerate(cidx):
            if c_ == chr_idx:
                left = i
                right = cidx.size
                break
        for i in range(left + 1, cidx.size):
            if c_ != chr_idx:
                right = i
                break
        # __find_range(cidx, chr_idx, rng)
        logger.info(f'{left}-{right}')
        n_cells = f5[f'{base}/bins/chrom'].size
        data = np.array(f5[f'{base}pixels/count'][left:right], dtype=np.int32)
        row = np.array(f5[f'{base}pixels/bin1_id'][left:right], dtype=np.int32)
        col = np.array(f5[f'{base}pixels/bin2_id'][left:right], dtype=np.int32)
        # print('binsize', binsize)
        mtx = scipy.sparse.coo_matrix((data, (row,col)), shape=(n_cells, n_cells), dtype=np.int32)
        f5.close()
        logger.info('sum={}'.format((np.sum(mtx))))
        
        hr = HiCSubregion(filename, chromosome, 1, chr_length, mtx, left, right, binsize)
    return hr

@numba.njit('void(i4[:],i4,i4[:])')
def __find_range(data, value, res):
    left = 0
    right = len(data)
    while left < right:
        center = (left + right) // 2
        val = data[center]
        if val < value:
            left = center + 1
        elif val > value:
            right = center
        else:
            break
    i = center
    start = 0
    while i >= 0:
        val = data[i]
        if val != value:
            start = i + 1
            break
        i -= 1
    end = len(data)
    i = center
    while i < len(data):
        val = data[i]
        if val != value:
            end = i
            break
        i += 1
    res[0] = start
    res[1] = end
    
def retrieve_interchromosome_matrix(filename:str, chromosome1:str, chromosome2:str, **kwargs)->HiCMatrix:
    """Chromosome 1 vs Chromosome 2 matrix.

    Args:
        filename (str): cool or mcool filename
        chromosome1 (str): first chromosome
        chromosome2 (str): second chromosome
        mode : 'raw' or 'weighted'
        logger : logging instance

    Raises:
        Exception: invalid chromosome

    Returns:
        HiCMatrix: matrix object
    """
    
    # determine binsize
    binsize = kwargs.get('resolution', 0)
    chroms = read_chromosomes(filename)
    chr_idx1 = chr_idx2 = -1
    resolutions = read_resolutions(filename)
    logger = kwargs.get('logger', logging.getLogger())
    mode = kwargs.get('mode', 'weighted')
    if binsize not in resolutions:
        binsize = resolutions[0]

    # convert chromosome name to internal index
    for i, c in enumerate(chroms.keys()):
        if chr_idx1 < 0:
            if chromosome1 == c:
                chr_idx1 = i
                if chr_idx2 >= 0: break
        if chr_idx2 < 0:
            if chromosome2 == c:
                chr_idx2 = i
                if chr_idx1 >= 0: break
    if chr_idx1 < 0 or chr_idx2 < 0: 
        sys.stderr.write(f'chromosome {chromosome1} or {chromosome2} is not included in {filename}\n')
        sys.stderr.write('available chromsoome names are {}\n'.format(list(chroms.keys())))
        raise Exception('invalid chromosome name')
    if chr_idx1 > chr_idx2:
        chr_idx1, chr_idx2 = chr_idx2, chr_idx1
        chromosome1, chromosome2 = chromosome2, chromosome1

    # set node base
    ft = determine_filetype(filename)
    if ft == 'mcool':
        base = f'resolutions/{binsize}/'
        pass
    elif ft == 'cool':
        base = ''
        pass
        
    with h5py.File(filename, 'r') as f5:
        cidx = np.array(f5[f'{base}bins/chrom'])

        rng1 = np.empty(2, dtype=np.int32)
        rng2 = np.empty(2, dtype=np.int32)
        __find_range(cidx, np.int32(chr_idx1), rng1)
        __find_range(cidx, np.int32(chr_idx2), rng2)

        weight1 = weight2 = None
        if mode == 'weighted':
            try:
                weights = np.array(f5[f'{base}bins/weight'])
                weight1 = weights[rng1[0]:rng1[1]]
                weight2 = weights[rng2[0]:rng2[1]]
            except:
                logger.info(f'{filename} is not balanced')
                pass
            
        counts = np.array(f5[f'{base}pixels/count'])
        rows = np.array(f5[f'{base}pixels/bin1_id'])
        cols = np.array(f5[f'{base}pixels/bin2_id'])
        # print('counts shape and chromosome range', counts.shape, rows.shape, cols.shape, rng1, rng2)
        mtx = scipy.sparse.coo_matrix((counts, (rows, cols)), dtype=np.int32)
        # print(mtx.shape)
        submat = mtx.tocsr()[rng1[0]:rng1[1],:].tocsc()[:,rng2[0]:rng2[1]]
        # print('result shape', submat.shape)
        f5.close()

        matrix = HiCMatrix(filename, chromosome1, chromosome2, submat, binsize, weight1, weight2)
        f5.close()
        return matrix


def retrieve_matrix_mcool(filename:str, positions:list, **kwargs)->list:
    """mcool to sparse matrix
    filename : mcool or cool
    positions : [ [chr, start,end], ...]
    resolution: mcool resolution (not used in cool processing)
    mode : 'raw' (raw count) or 'weighted' (normalized by cooler)
    logger: log output
    """
    # print(kwargs)
    logger = kwargs.get('logger', logging.getLogger())
    node = kwargs.get('node', None)
    resolution = kwargs.get('resolution', 0)
    matrices = []
    mode = kwargs.get('mode', 'weighted')
    # print(mode)
    with h5py.File(filename) as f5:
        if node is None:
            logger.info('mcool')
            resolutions = []
            for resol in f5['resolutions'].keys():
                if resol.isdigit():
                    resolutions.append(int(resol))
            if len(resolutions) == 0:
                raise Exception('no hic resolutions')
            if resolution in resolutions:
                binsize = resolution
            else:
                binsize = min(resolutions)
            logger.info(f'adjust bin size as {binsize}')
            node = f'resolutions/{binsize}/'
        logger.info(f'processing node {node}')
        mtx = None        
        chromosomes = [_.decode('utf-8') for _ in f5[f'{node}/chroms/name']]
        lengths = f5[f'{node}chroms/length']
        n_chrom = len(chromosomes)

        bin_chr   = list(f5[f'{node}bins/chrom'])
        bin_start = list(f5[f'{node}bins/start'])
        bin_end   = list(f5[f'{node}bins/end'])
        chr_bin = -1
        n_bins = len(bin_chr)
        logger.info('scanning {} bins'.format(n_bins))
        for chrom, start, end in positions:
            logger.info('{}:{}-{}'.format(chrom, start, end))
            chr_bin = -1
            idx_start = idx_end = -1
            for i, c in enumerate(chromosomes):
                if chrom == c or re.sub('^chr', '', chrom) == re.sub('^chr', '', c):
                    logger.info(f'index of {chrom} is {i}:{c}')
                    chr_bin = i
                    break
            left = 0
            right = n_bins
            index = -1
            while left < right:
                i = (left + right) // 2
                c_ = bin_chr[i]
                s_ = bin_start[i]
                e_ = bin_end[i]
                if c_ < chr_bin or (c_ == chr_bin and e_ < start):
                    left = i + 1
                elif c_ > chr_bin or (c_ == chr_bin and end < s_):
                    right = i
                else:
                    index = i
                    break
            if index >= 0:
                logger.info(f'scanning from {index}')
                i = index
                index_start = 0
                index_end = n_bins
                matrix_start = matrix_end = -1
                while i >= 0:
                    c_ = bin_chr[i]
                    s_ = bin_start[i]
                    e_ = bin_end[i]
                    if c_ != chr_bin or s_ <= start:
                        index_start = i
                        matrix_start = s_
                        break
                    i -= 1
                i = index
                while i < n_bins:
                    c_ = bin_chr[i]
                    s_ = bin_start[i]
                    e_ = bin_end[i]
                    if c_ != chr_bin or end <= e_:
                        index_end = i + 1
                        matrix_end = e_
                        break
                    i += 1
                logger.info(f'matrix range : {index_start}-{index_end}')
                if mtx is None: 
                    rows = f5[f'{node}pixels/bin1_id']
                    cols = f5[f'{node}pixels/bin2_id']
                    count = f5[f'{node}pixels/count']
                    logger.info('loading matrix')
                    mtx = scipy.sparse.coo_matrix((count, (rows, cols)), dtype=np.int32).tocsr()
                    logger.info('loaded matrix {}x{} having {} counts'.format(mtx.shape[0], mtx.shape[1], mtx.sum()))
                submat = mtx[index_start:index_end, index_start:index_end].toarray().astype(np.float32)
                # set weight if exists
                region = HiCSubregion(filename, chrom, matrix_start, matrix_end, submat, index_start, index_end)#mtx[index_start:index_end, index_start:index_end].toarray().astype(np.float32))
                if not rawcount_mode:
                    try:
                        logger.info(f'loading weight from {node}bins/weight')
                        weight = list(f5[f'{node}bins/weight'])
                        logger.info('{}-{} of {} weight array'.format(index_start, index_end, len(weight)))
                        region.set_weight(np.array(weight[index_start:index_end], dtype=np.float32))
                    except Exception as e:
                        logger.warning(str(e))
                matrices.append(region)
        f5.close()
    return matrices

def read_chromosomes(filename:str, **kwargs)->collections.OrderedDict:
    """"
    Args:
        filename (str): cool/mcool filename

    Returns:
       OrderedDict : chromosome as key, length as value
    """
    resolution = kwargs.get('resolution', 0)
    ftype = determine_filetype(filename)
    if ftype == 'mcool':
        res = read_resolutions(filename)
        if resolution not in res:
            resolution = min(res)
        node = 'resolutions/{}/'.format(resolution)
    else:
        node = ''
    with h5py.File(filename, 'r') as f5:
        chroms   = list(f5[f'{node}chroms/name'])
        lengths = list(f5[f'{node}chroms/length'])

        chromosomes = collections.OrderedDict(zip([_.decode('utf-8') for _ in chroms], lengths))
        f5.close()
        return chromosomes

def show_tree(filename, ostr=sys.stdout):#logger=None):
    def __visitor_func(name, node):
        ostr = sys.stdout
        attr = node.attrs
        ostr.write('{}\tis_data:{}\t{}\n'.format(name, isinstance(node, h5py.Dataset), node))
        if attr is not None and len(attr) > 0:
            astr = ''
            for key in attr:
                astr += '{}={},'.format(key, attr[key])
            ostr.write('\t4 attributes : {}\n'.format(astr))
        return None

    with h5py.File(filename, 'r') as f:
        f.visititems(__visitor_func)
    pass

def display_tad():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('-b', type=int, default=0, help='binsize')
    parser.add_argument('-i', metavar='filename')
    parser.add_argument('-o', default='hic_out', metavar='directory')                    
    parser.add_argument('-p', default=['chr1:1-1000000', ], nargs='+')
    parser.add_argument('--level', help='chart intensity (value:max value, percentile(p=X):X oercentile)', default='p:99.9')
    
    args, cmds = parser.parse_known_args()
    
    logger = tkutil.get_logger(__name__)
    rawmode = args.raw
    if verbose: logger.setLevel(10)

    # for c in cmds:
    #     if c in available_commands:
    #         if c not in commands:
    #             commands.append(c)
    #     elif os.path.exists(c) and (c.endswith('.cool') or c.ends('.mcool')):
    #         filenames.append(c)
    filename = args.i
    
    
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('-b', type=int, default=0, help='binsize')
    parser.add_argument('-i', metavar='filename')
    parser.add_argument('-o', default='hic_out', metavar='directory')                    
    parser.add_argument('-p', default=['chr1:1-1000000', ], nargs='+', help='location')
    parser.add_argument('--level', help='chart intensity (value:max value, percentile(p=X):X oercentile)', default='p:99.9')
    
    args, cmds = parser.parse_known_args()
    if len(cmds) == 0:
        cmds = ['ls', ]
    commands = []
    verbose = args.verbose
    available_commands = ['ls', 'chr']
    filename = args.i
    positions = args.p
    outdir = args.o
    filenames = [args.i, ]
    logger = tkutil.get_logger(os.path.basename(__file__))
    rawmode = args.raw
    if verbose: logger.setLevel(10)

    for c in cmds:
        if c in available_commands:
            if c not in commands:
                commands.append(c)
        elif os.path.exists(c) and (c.endswith('.cool') or c.endswith('.mcool')):
            filenames.append(c)
    filename = filenames[0]

    # threshold
    level = args.level
    threshold = percentile = 0
    m = re.search('(\\w+)=([\\d\\.]+)', level)
    if m:
        if m.group(1) in ['p', 'percentile', 'pt']:
            percentile = float(m.group(2))
            threshold = 0
        elif m.group(1) in ['t', 'thr', 'threshold']:
            threshold = float(m.group(2))
            percentile = 0
    
    for cmd in cmds:
        logger.info('{}\t{}'.format(cmd, filename))
        if cmd == 'ls' or cmd == 'list':
            show_tree(filename)
        elif cmd == 'tad':
            display_tad()
        elif cmd == 'chr':
            for chrom, length in read_chromosomes(filename).items():
                print('{}\t{}'.format(chrom, length))
        elif cmd == 'intra':
            chroms = []
            for p in positions:
                chrom = p.split(':')[0]
                chroms.append(p)
                logger.info(f'{chrom}')
                mtx = retrieve_chromosome_matrix(filename, chrom, logger=logger)
                # print(mtx.shape)
                fn_out = os.path.join(outdir, chrom + '.hicmat')
                if fn_out is not None:
                    mtx.dump(fn_out)
                img = plot_matrix(mtx, rawmode=rawmode, logger=logger, percentile=percentile, threshold=0 )      
                fn_img = os.path.join(outdir, 'tad_{}.png'.format(mtx.chromosome))
                skimage.io.imsave(fn_img, img)
        elif cmd == 'inter':
            os.makedirs(outdir, exist_ok=1)
            chroms = []
            for p in positions:
                chrom = p.split(':')[0]
                chroms.append(p)
            if len(chroms) == 1:
                chroms = [chroms[0], chroms[0]]
            mtx = retrieve_interchromosome_matrix(filename, chroms[0], chroms[1], logger=logger)
            chrom_mat = mtx.chromosomes
            fn_img = os.path.join(outdir, 'image_{}_{}.png'.format(chrom_mat[0], chrom_mat[1]))
            # print('matrix type', type(mtx))
            img = plot_matrix(mtx, threshold=threshold, percentile=percentile, logger=logger, rawmode=rawmode)
            skimage.io.imsave(fn_img, img)
            pass
    pass

if __name__ == '__main__':
    main()
