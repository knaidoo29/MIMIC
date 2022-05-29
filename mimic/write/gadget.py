import struct
import numpy as np


def write_gadget(outfile, header, pos, vel, ic_format=2, single=False, id_offset=0):
    """Write binary initial condition (ICs) files for GADGET and AREPO for dark
    matter only simulations.

    Parameters
    ----------
    outfile : str
        Output filename of the initial conditions. If single is True, the
        outfile string is used directly to name the ICs. If single is False,
        the outfile string is appended with the file number, e.g. file.0,
        file.1, etc.
    header : dict
        Dictionary that contains all of the necessary header information.
    pos : (N, 3) array
        Array containing the particle positions.
    vel : (N, 3) array
        Array containing the particle velocities
    ic_format : (1, 2) int
        Generate initial condition with either type 1 or type 2 binary Gadget
        format.
    single : boolean
        If single is True, all particle are saved in a single file, even if
        nfiles!=1 in the header. This can be used to save particle data that
        is too large for memory by chuncking the data before passing it into
        the function.
    id_offset : int
        Offset that is added to the particle IDs. Primarily used when single
        is True.
    """

    # Header dictionary keys and data types
    # Ordering is important here!
    h = header.copy()
    h_keys = ['npart', 'massarr', 'time', 'redshift', 'flag_sfr',
              'flag_feedback', 'npart_all', 'flag_cooling', 'nfiles',
              'boxsize', 'omegam', 'omegal', 'hubble', 'flag_age',
              'flag_metals', 'npart_all_hw']
    h_dtype = [np.uint32, np.float64, np.float64, np.float64, np.int32,
               np.int32, np.uint32, np.int32, np.int32, np.float64,
               np.float64, np.float64, np.float64, np.int32, np.int32,
               np.uint32]

    # Basic checks of input data
    assert pos.shape[1] == 3
    assert vel.shape[1] == 3
    assert pos.shape[0] == vel.shape[0]
    if single:
        assert h['npart_all'] != 0
        assert np.issubdtype(type(id_offset), np.integer)
    else:
        assert pos.shape[0] == h['npart_all']

    # Fill in header information
    if single:
        npart_all = int(h['npart_all'])
        h['npart_all'] = [0, npart_all, 0, 0, 0, 0]
    else:
        npart_all = pos.shape[0]
        h['npart_all'] = [0, npart_all, 0, 0, 0, 0]
    h['npart_all_hw'] = [0, 0, 0, 0, 0, 0]
    h['massarr'] = [0, h['massarr'], 0, 0, 0, 0]
    h['flag_sfr'] = 0
    h['flag_feedback'] = 0
    h['flag_cooling'] = 0
    h['flag_age'] = 0
    h['flag_metals'] = 0

    # Check if total number of particles exceeds 32 bytes
    longids = False
    if np.any(npart_all >= 2**32):
        longids = True

    # Split the total number of particles bitwise into 2 uint32 values
    if longids:
        npart_i64 = struct.pack('L', np.uint64(npart_all))
        h['npart_all'][1] = struct.unpack('I', npart_i64[:4])[0]
        h['npart_all_hw'][1] = struct.unpack('I', npart_i64[4:])[0]

    # Calculate an even split of particles across multiple files
    if not single:
        nfiles = h['nfiles']
        npart_arr = np.repeat(int(npart_all/nfiles), nfiles)
        npart_arr = [n+int(i<npart_all%nfiles) for i,n in enumerate(npart_arr)]
        arr_split = np.hstack([0, np.cumsum(npart_arr)])

    # Create a list IC file names
    if single:
        outfiles = [outfile]
    else:
        outfiles = [outfile+'.'+str(nf) for nf in range(h['nfiles'])]

    for i, of in enumerate(outfiles):
        with open(of, 'wb') as f:
            # Write header
            if ic_format == 2:
                f.write(struct.pack('i', np.int32(8)))
                f.write(b'HEAD')
                f.write(struct.pack('i', np.int32(264)))
                f.write(struct.pack('i', np.int32(8)))
            f.write(struct.pack('i', np.int32(256)))

            if single:
                npart = pos.shape[0]
            else:
                npart = npart_arr[i]
            h['npart'] = [0, npart, 0, 0, 0, 0]
            for ki, key in enumerate(h_keys):
                attr = np.array([h[key]], dtype=h_dtype[ki])
                f.write(bytes(attr))

            # Fill the rest of the header block up to 256 bytes with zeros
            f.write(bytes(np.zeros(16, dtype=np.int32)))

            f.write(struct.pack('i', np.int32(256)))

            # Write particle positions
            if ic_format == 2:
                f.write(struct.pack('i', np.int32(8)))
                f.write(bytes(b'POS '))
                f.write(struct.pack('i', np.int32(npart*12+8)))
                f.write(struct.pack('i', np.int32(8)))
            f.write(struct.pack('i', np.int32(npart*12)))

            if single:
                f.write(bytes(pos.flatten().astype(np.float32)))
            else:
                pos_file = pos[arr_split[i]:arr_split[i+1]]
                pos_file = pos_file.flatten()
                f.write(bytes(pos_file.astype(np.float32)))

            f.write(struct.pack('i', np.int32(npart*12)))

            # Write particle velocities
            if ic_format == 2:
                f.write(struct.pack('i', np.int32(8)))
                f.write(bytes(b'VEL '))
                f.write(struct.pack('i', np.int32(npart*12+8)))
                f.write(struct.pack('i', np.int32(8)))
            f.write(struct.pack('i', np.int32(npart*12)))

            if single:
                f.write(bytes(vel.flatten().astype(np.float32)))
            else:
                vel_file = vel[arr_split[i]:arr_split[i+1]]
                vel_file = vel_file.flatten()
                f.write(bytes(vel_file.astype(np.float32)))

            f.write(struct.pack('i', np.int32(npart*12)))

            # Write particle ids
            if longids:
                id_byte_size = 8
                id_dtype = np.uint64
            else:
                id_byte_size = 4
                id_dtype = np.uint32

            if ic_format == 2:
                f.write(struct.pack('i', np.int32(8)))
                f.write(bytes(b'ID  '))
                f.write(struct.pack('i', np.int32(npart*id_byte_size+8)))
                f.write(struct.pack('i', np.int32(8)))
            f.write(struct.pack('i', np.int32(npart*id_byte_size)))

            if not single:
                id_offset = np.sum(arr_split[:i+1])
            ids = np.arange(npart) + id_offset
            f.write(bytes(ids.astype(id_dtype)))

            f.write(struct.pack('i', np.int32(npart*id_byte_size)))
