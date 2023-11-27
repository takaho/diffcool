from coolhandler import *
import cellrangerwrapper
import tkutil
import argparse
import os, sys, re, io, json
import scipy.io

def save_sparse(filename, mtx):
    scipy.io.mmwrite(filename, mtx)
#def save_tsv(filename, mtx):
    
def main():
    logger = tkutil.get_logger(os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+', metavar='mcool files', help='mcool files')
    parser.add_argument('-c', metavar='control mcool files', help='mcool file')
    parser.add_argument('-t', metavar='treated mcool files', help='mcool file')
    parser.add_argument('-o', default='difout', metavar='directory', help='output directory')
    parser.add_argument('--resolution', default=0, help='resolution')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    filenames = args.i
    if args.c is not None:
        filename_control = args.c
    else:
        filename_control = filenames[0]
    if args.t is not None:
        filename_traeted = args.t
    else:
        filename_treated = filenames[0]

    filenames = [filename_control, filename_treated]
    
    if args.verbose:
        logger.setLevel(10)
        pass
    outdir = args.o
    info = {
        'control':filename_control,
        'treated':filename_treated,
        }
    os.makedirs(outdir, exist_ok=1)
    
    chromosomes = []
    resolution_sets = set()
    for fn in filenames:
        logger.info('reading resolution from {}'.format(fn))
        res = read_resolutions(fn)
        if len(resolution_sets) == 0:
            resolution_sets = set(res)
        else:
            resolution_sets = set([r for r in res if r in resolution_sets])
        for c in read_chromosomes(fn):#chromosomes:
            if c not in chromosomes:
                chromosomes.append(c)
                pass
            pass
        pass
    if len(chromosomes) == 0 or len(resolution_sets) == 0:
        raise Exception('common chromoisomes and common resolutions are required')
    n_cool = len(filenames)
    info['resolutions'] = [int(x_) for x_ in resolution_sets]
    info['chromosomes'] = list(chromosomes)
    print(json.dumps(info, indent=2))
    #fn_outcool_1 = os.path.join(outdir, 'ratio.

    fn_out_diff = os.path.join(outdir, 'diff.mcool')
    fn_out_ratio = os.path.join(outdir, 'ratio.mcool')

    info['diff'] = fn_out_diff
    info['ratio'] = fn_out_ratio
    
    fn_info = os.path.join(outdir, 'run.info')
    with open(fn_info, 'w') as fo:
        json.dump(info, fo, indent=2)
        pass
    """
resolutions/8192000	is_data:False	<HDF5 group "/resolutions/8192000" (4 members)>
resolutions/8192000/bins	is_data:False	<HDF5 group "/resolutions/8192000/bins" (4 members)>
resolutions/8192000/bins/chrom	is_data:True	<HDF5 dataset "chrom": shape (343,), type "<i4">
resolutions/8192000/bins/end	is_data:True	<HDF5 dataset "end": shape (343,), type "<i4">
resolutions/8192000/bins/start	is_data:True	<HDF5 dataset "start": shape (343,), type "<i4">
resolutions/8192000/bins/weight	is_data:True	<HDF5 dataset "weight": shape (343,), type "<f8">
resolutions/8192000/chroms	is_data:False	<HDF5 group "/resolutions/8192000/chroms" (2 members)>
resolutions/8192000/chroms/length	is_data:True	<HDF5 dataset "length": shape (22,), type "<i4">
resolutions/8192000/chroms/name	is_data:True	<HDF5 dataset "name": shape (22,), type "|S5">
resolutions/8192000/indexes	is_data:False	<HDF5 group "/resolutions/8192000/indexes" (2 members)>
resolutions/8192000/indexes/bin1_offset	is_data:True	<HDF5 dataset "bin1_offset": shape (344,), type "<i8">
resolutions/8192000/indexes/chrom_offset	is_data:True	<HDF5 dataset "chrom_offset": shape (23,), type "<i8">
resolutions/8192000/pixels	is_data:False	<HDF5 group "/resolutions/8192000/pixels" (3 members)>
resolutions/8192000/pixels/bin1_id	is_data:True	<HDF5 dataset "bin1_id": shape (57942,), type "<i8">
resolutions/8192000/pixels/bin2_id	is_data:True	<HDF5 dataset "bin2_id": shape (57942,), type "<i8">
resolutions/8192000/pixels/count	is_data:True	<HDF5 dataset "count": shape (57942,), type "<i4">
"""

    def get_chromosome_set(filenames, resolution=None)->list:
        chromosomes = []
        lengths = []
        if resolution is None: # cool
            base = ''
        else:
            base = f'resolutions/{resolution}'
        for fn in filenames:
            with h5py.File(fn) as fh:
                chrom_ = fh[f'{base}bin/chroms/name']
                length_ = fh[f'{base}bin/chroms/name']
                if len(chromosomes) == 0:
                    chromosomes = chrom_
                    lengths = length_
                else:
                    c_ = []
                    l_ = []
                    for i, c in enumerate(chrom_):
                        if c in chromosomes and c not in c_:
                            c_.append(c)
                            l_.append(lengths[i])
                    chromosomes = c_
                    lengths = l_
        return zip(chromosomes, lengths)
                
    chromosomes = get_chromosome_set([filename_control, filename_treated], max(resolution_sets))
    print(chromosomes)
    
    with h5py.File(fn_out_diff, 'w') as fo1, h5py.File(filename_control) as fcnt, h5py.File(filename_treated) as ftrt:
        for res in sorted(resolution_sets, reverse=True):
            chr_idx = 0
            # sparse matrix for all chromosomes
            mtx_rows = []
            mtx_cols = []
            mtx_data = []
            chr_idx = []
            chr_idx = 0
            offset = 0
            chr_start = []
            chr_end = []
            for chromosomes, length in chromosomes:
                chrom_offset.append(offset)
                
                fo1.create_dataset(f'resolutions/{res}/bins/chroms/length', chrom_lengths, dtype=np.int32)
                fo1.create_dataset(f'resolutions/{res}/bins/chroms/name', chromosomes)

                cnt = retrieve_chromosome_matrix(filename_control, chrom, resolution=resolution, logger=loggger).tocoo()
                trt = retrieve_chromosome_matrix(filename_control, chrom, resolution=resolution, logger=loggger).tocoo()

                size = min(cnt.shape[0], trt.shape[0])

#                matrix = scipy.sparse.lil_matrix((size, size), dtype=np.int4)
                
                n_cnt = len(cnt.data)
                n_trt = len(trt.data)

                diff = trt - cnt
                n_diff = len(diff.data)

                idx_start = len(mtx_data)
                mtx = matrix.tocoo()
                mtx_rows += diff.row
                mtx_cols += diff.col
                mtx_data += diff.data
                idx_end = len(mtx_data)

                bin1_offset.append([chr_idx] * size)
                chr_idx.append([chr_idx] * size)
                for i in range(size):
                    chr_start.append(i * binsize)
                    chr_end.append((i + 1) * binsize)
                offset += size

                
            #fo1.create_dataset(f'/resolutions/{res}/bins/weight', chrom_lengths, dtype=np.int32)
            fo1.create_dataset(f'/resolutions/{res}/bins/chrom', chridx, dtype=np.int32)
            fo1.create_dataset(f'/resolutions/{res}/bins/end', endidx, dtype=np.int32)
            fo1.create_dataset(f'/resolutions/{res}/bins/start', startidx, dtype=np.int32)

            # row index for each chromosome and chromosome index for each row
            chrom_offset = []
            bin1_offset = []
            fo1.create_dataset(f'/resolutions/{res}/bins/indexes/bin1_offset', bin1_offset, dtype=np.int64) # row index for chromosomome start
            fo1.create_dataset(f'/resolutions/{res}/bins/indexes/chrom_offset', chrom_offset, dtype=np.int64) # row index for each chromosome

            fo1.create_dataset(f'/resolutions/{res}/pixels/bin1_id', mtx_rows, dtype=np.int64)
            fo1.create_dataset(f'/resolutions/{res}/pixels/bin2_id', mtx_cols, dtype=np.int64)
            fo1.create_dataset(f'/resolutions/{res}/pixels/count', mtx_data, dtype=np.int32)
            
            #fo1.create_dataset(f'/resolutions/{res}/bins/indexes/chrom_offset', chrom_offset, dtype=np.int64) # row index for each chromosome
            
            # sparse matrix for all chromosomes
            mtx_rows = []
            mtx_cols = []
            mtx_vals = []

            chr_ids = []
            
            for i in range(n_chroms):
                cnames_.append(chromosomes[i])
                clengths_.append(lengths[i])

                # process chromosome
                ind_start = len(vals)
                # control, treated
                vals_ = []
                rows_ = []
                cols_ = []
                if 0:
                    vals_.append(v)
                    rows_.append(r)
                    cols_.append(c)

                rows += rows_
                cols += cols_
                vals += vals_
                
                ind_end = len(vals)

                
            fo1.create_dataset(f'/resolutions/{res}/bins/indexes/bin1_offset', bin1_offset, dtype=np.int64)
            fo1.create_dataset(f'/resolutions/{res}/bins/indexes/chrom_offset', chrom_offset, dtype=np.int64)
                

            fo1.create_dataset(f'/resolutions/{res}/bins/pixels/bin1_id', rows, dtype=np.int64)
            fo1.create_dataset(f'/resolutions/{res}/bins/pixels/bin2_id', cols, dtype=np.int64)
            fo1.create_dataset(f'/resolutions/{res}/bins/pixels/count', vals, dtype=np.int32)
            
        

        
    for res in sorted(resolution_sets, reverse=True):
        logger.info('resolution={}'.format(res))
        for chrom in chromosomes:
            countdata = []
            logger.info('chromosomes={}'.format(chrom))
            for i, fn in enumerate(filenames):
                mtx = retrieve_chromosome_matrix(fn, chrom, resolution=res, logger=logger)
                data = mtx.data#.toarray()
                logger.info(f'{chrom}\t{i}\t{data.shape}')
                countdata.append(data)
                pass
            # calculate differentiation
            for i in range(n_cool):
                ci = countdata[i]
                print(ci)
                for j in range(i + 1, n_cool):
                    cj = countdata[j]
                    diff = ci - cj
                    plus = ci + cj
                    ratio = diff / plus#diff / (ci + cj + 1)
                    # saving to file
                    logger.info('saving {} resolution {} : {} / {}'.format(chrom, res, i, j))
                    fn_out = os.path.join(outdir, 'diff_{}_r{}__{}_{}.mtx'.format(chrom, res, i, j))
                    save_sparse(fn_out, diff)
                    #np.savetxt(fn_out, diff, format='{:.3f}', delimiter='\t')
                    fn_out = os.path.join(outdir, 'ratio_{}_r{}__{}_{}.mtx'.format(chrom, res, i, j))
                    #np.savetxt(fn_out, ratio, format='{:.3f}', delimiter='\t')
                    save_sparse(fn_out, ratio)
                    
                    
        
        
    
if __name__ == '__main__':
    main()

