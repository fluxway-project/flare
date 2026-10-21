# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated line parsing.

C-level string scanning (memchr) + number conversion (strtod/strtol).
Hot loop writes directly into preallocated numpy buffers via typed
memoryviews -- zero per-row PyObject allocation, zero per-line bytes copy.

API:
    parse_lines_fast(lines, code_to_base_meta,
                     direct and metadata column indices,
                     metadata maps, feature_buffers, meta_ints)
    -> (feature_buffers, meta_ints, batch_size)

Caller (RowParser) pre-allocates:
    feature_buffers[name] : np.ndarray (int64 or float32)
        - scalar feature: shape (batch,)
        - vector feature: shape (batch, length)
        - prefilled with pad_val (-3 / -3.0)
    meta_ints[name]       : np.ndarray int32 (batch,)  -- parser metadata fields
"""

from libc.stdint cimport int32_t, uint8_t
from libc.stdlib cimport strtod
from libc.string cimport memchr, memcpy

cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t* size) except NULL
    object PyUnicode_FromStringAndSize(const char* u, Py_ssize_t size)

cdef int TAB = 0x09        # '\t'
cdef int MAP_SEP = 0x02    # '\x02'
cdef int KV_SEP = 0x03     # '\x03'
cdef int OPEN_BRACKET = 0x5B   # '['
cdef int CLOSE_BRACKET = 0x5D  # ']'
cdef int COMMA = 0x2C          # ','
cdef int MINUS = 0x2D          # '-'
cdef int PLUS = 0x2B           # '+'

cdef inline bint _scalar_is_present(const char* start, const char* end) noexcept nogil:
    return start < end and not (end - start == 1 and start[0] == 0x30)


cdef inline double _fast_atof(const char* s, const char* end) noexcept nogil:
    """Parse float from a bounded C string without creating Python objects."""
    cdef char buf[64]
    cdef Py_ssize_t n = end - s
    if n <= 0 or n >= 63:
        return -3.0
    memcpy(buf, s, n)
    buf[n] = 0
    return strtod(buf, NULL)


cdef inline long _slow_atol(const char* s, const char* end) noexcept nogil:
    """Preserve strtod's permissive conversion for non-integer tokens."""
    cdef char buf[64]
    cdef Py_ssize_t n = end - s
    if n <= 0 or n >= 63:
        return -3
    memcpy(buf, s, n)
    buf[n] = 0
    return <long>strtod(buf, NULL)


cdef inline long _fast_atol(const char* s, const char* end) noexcept nogil:
    """Parse common short decimal integers without copying or ``strtod``.

    The source can also contain forms such as ``1.0``. Only an optional sign
    followed by at most 15 decimal digits takes this path; every other token
    deliberately retains the former bounded ``strtod`` behaviour.
    """
    cdef const char* p = s
    cdef Py_ssize_t digits
    cdef long value = 0
    cdef int digit
    cdef bint negative = False

    if p >= end:
        return -3
    if p[0] == MINUS:
        negative = True
        p += 1
    elif p[0] == PLUS:
        p += 1
    digits = end - p
    if digits <= 0 or digits > 15:
        return _slow_atol(s, end)
    while p < end:
        digit = p[0] - 0x30
        if digit < 0 or digit > 9:
            return _slow_atol(s, end)
        value = value * 10 + digit
        p += 1
    return -value if negative else value


cdef int _parse_vector_int_ptr(
    long* ptr,
    const char* val_start,
    const char* val_end,
    int length,
) noexcept nogil:
    """Parse a comma-separated integer vector into ptr[:count]. Returns count."""
    cdef const char* p = val_start
    cdef const char* comma_ptr
    cdef int count = 0
    cdef long int_val

    while p < val_end and count < length:
        comma_ptr = <const char*>memchr(p, COMMA, val_end - p)
        if comma_ptr == NULL:
            comma_ptr = val_end
        
        int_val = _fast_atol(p, comma_ptr)
        if int_val != -3 or (comma_ptr - p == 2 and p[0] == MINUS):
            ptr[count] = int_val
            count += 1
            
        p = comma_ptr + 1
    return count


cdef int _parse_vector_float_ptr(
    float* ptr,
    const char* val_start,
    const char* val_end,
    int length,
) noexcept nogil:
    """Parse a comma-separated float vector into ptr[:count]. Returns count."""
    cdef const char* p = val_start
    cdef const char* comma_ptr
    cdef int count = 0
    cdef double float_val

    while p < val_end and count < length:
        comma_ptr = <const char*>memchr(p, COMMA, val_end - p)
        if comma_ptr == NULL:
            comma_ptr = val_end
            
        float_val = _fast_atof(p, comma_ptr)
        if float_val != -3.0 or (comma_ptr - p == 4 and p[0] == MINUS):
            ptr[count] = <float>float_val
            count += 1
            
        p = comma_ptr + 1
    return count


def parse_arrow_direct(
    const char[::1] data_buf,
    const long[::1] offsets,
    Py_ssize_t n_rows,
    dict code_to_base_meta,
    tuple direct_columns,
    int required_col_count,
    int payload_col_idx,
    tuple metadata_columns,
    dict metadata_maps,
    object feature_buffers,
    dict meta_ints,
):
    """Zero-copy Arrow buffer → parsed feature buffers.

    Reads directly from the Arrow LargeStringArray's contiguous data buffer
    and int64 offsets array — no intermediate Python str objects.

    Args:
        data_buf: Arrow values buffer (contiguous UTF-8 bytes) as memoryview.
        offsets: Arrow offsets buffer (int64, length = n_rows + 1) as memoryview.
        n_rows: Number of rows to parse.
        code_to_base_meta: {int_code: (feat_name, length, is_float, pad_val)}.
        direct_columns: ordered ``(name, column_index)`` scalar columns.
        required_col_count: minimum number of tab-separated columns.
        payload_col_idx: source column containing the feature payload.
        metadata_columns: ordered ``(name, column_index)`` metadata columns.
        metadata_maps: ``{name: {raw_value: encoded_value}}`` mappings.
        feature_buffers: {feat_name: np.ndarray} -- pre-allocated buffers.
        meta_ints: parser metadata fields mapped to int32 arrays.

    Returns:
        (feature_buffers, meta_ints, n_rows).
    """
    cdef const char* data_ptr = <const char*>&data_buf[0]
    cdef const long* off_ptr = &offsets[0]
    cdef Py_ssize_t batch_size = n_rows
    cdef Py_ssize_t row_idx, n_bytes
    cdef const char* buf
    cdef const char* buf_end
    cdef const char* col_starts[64]
    cdef const char* col_ends[64]
    cdef const char* p
    cdef const char* seg_start
    cdef const char* seg_end
    cdef const char* seg_end_ptr
    cdef const char* kv_ptr
    cdef const char* val_start
    cdef const char* val_end
    cdef int col_count
    cdef long code_int
    cdef int length_i
    cdef bint is_float_i
    cdef double float_val
    cdef long int_val
    cdef Py_ssize_t metadata_off, metadata_len
    cdef object metadata_name, metadata_value, metadata_map

    cdef long code_iter
    cdef int len_iter
    cdef bint is_float_iter

    # LUTs for fast code_int resolution
    cdef bint lut_valid[65536]
    cdef int lut_length[65536]
    cdef bint lut_is_float[65536]
    cdef float* sf_ptrs[65536]
    cdef long* si_ptrs[65536]
    cdef float* vf_ptrs[65536]
    cdef long* vi_ptrs[65536]
    cdef int* vo_ptrs[65536]
    cdef uint8_t* scalar_valid_ptrs[65536]
    cdef int vo_counters[65536]

    cdef long vector_codes[2048]
    cdef int num_vector_codes = 0
    
    import numpy as np
    cdef int i
    for i in range(65536):
        lut_valid[i] = False
        vo_counters[i] = 0

    cdef object scalar_validity = getattr(feature_buffers, "scalar_validity")
    cdef list direct_names = [item[0] for item in direct_columns]
    cdef list direct_indices = [item[1] for item in direct_columns]
    cdef list direct_buffers = [feature_buffers[name] for name in direct_names]
    cdef list direct_validity = [scalar_validity[name] for name in direct_names]

    cdef long[::1] si_view
    cdef float[::1] sf_view
    cdef long[::1] vi_view
    cdef float[::1] vf_view
    cdef int[::1] vo_view
    cdef uint8_t[::1] scalar_valid_view

    for code_iter, meta_tuple in code_to_base_meta.items():
        if code_iter < 0 or code_iter >= 65536:
            continue
        feat_name = meta_tuple[0]
        if feat_name in direct_names:
            continue
        len_iter = meta_tuple[1]
        is_float_iter = meta_tuple[2]
        buf_obj = feature_buffers[feat_name]
        
        lut_valid[code_iter] = True
        lut_length[code_iter] = len_iter
        lut_is_float[code_iter] = is_float_iter
        
        if len_iter == 1:
            if is_float_iter:
                sf_view = buf_obj
                sf_ptrs[code_iter] = &sf_view[0]
            else:
                si_view = buf_obj
                si_ptrs[code_iter] = &si_view[0]
            scalar_valid_view = scalar_validity[feat_name]
            scalar_valid_ptrs[code_iter] = &scalar_valid_view[0]
        else:
            flat_buf, off_buf = buf_obj
            if is_float_iter:
                vf_view = flat_buf
                vf_ptrs[code_iter] = &vf_view[0]
            else:
                vi_view = flat_buf
                vi_ptrs[code_iter] = &vi_view[0]
            vo_view = off_buf
            vo_ptrs[code_iter] = &vo_view[0]
            
            vector_codes[num_vector_codes] = code_iter
            num_vector_codes += 1

    for row_idx in range(batch_size):
        buf = data_ptr + off_ptr[row_idx]
        n_bytes = off_ptr[row_idx + 1] - off_ptr[row_idx]
        buf_end = buf + n_bytes

        if required_col_count > 64 or payload_col_idx < 0 or payload_col_idx >= required_col_count:
            raise ValueError("parser requires more than 64 source columns")
        # Split source columns by finding tabs.
        col_starts[0] = buf
        col_count = 0
        p = buf
        while p < buf_end and col_count < required_col_count - 1:
            p = <const char*>memchr(p, TAB, buf_end - p)
            if p == NULL:
                break
            col_ends[col_count] = p
            col_count += 1
            p += 1
            col_starts[col_count] = p

        if col_count != required_col_count - 1:
            raise ValueError(f"Expected {required_col_count} tab-separated columns, got {col_count + 1}")
        col_ends[required_col_count - 1] = buf_end

        for i in range(len(direct_names)):
            direct_buffers[i][row_idx] = _fast_atol(col_starts[direct_indices[i]], col_ends[direct_indices[i]])
            direct_validity[i][row_idx] = _scalar_is_present(col_starts[direct_indices[i]], col_ends[direct_indices[i]])

        for i in range(len(metadata_columns)):
            metadata_name = metadata_columns[i][0]
            metadata_off = col_starts[metadata_columns[i][1]] - buf
            metadata_len = col_ends[metadata_columns[i][1]] - col_starts[metadata_columns[i][1]]
            metadata_value = PyUnicode_FromStringAndSize(buf + metadata_off, metadata_len)
            metadata_map = metadata_maps[metadata_name]
            try:
                meta_ints[metadata_name][row_idx] = metadata_map[metadata_value]
            except KeyError as error:
                raise KeyError(f"unknown metadata value for {metadata_name}: {metadata_value!r}") from error

        # Parse the explicitly configured feature payload column.
        seg_start = col_starts[payload_col_idx]
        seg_end = col_ends[payload_col_idx]
        if seg_start >= seg_end:
            continue

        p = seg_start
        while p < seg_end:
            seg_end_ptr = <const char*>memchr(p, MAP_SEP, seg_end - p)
            if seg_end_ptr == NULL:
                seg_end_ptr = seg_end

            if seg_end_ptr == p:
                p = seg_end_ptr + 1
                continue

            kv_ptr = <const char*>memchr(p, KV_SEP, seg_end_ptr - p)
            if kv_ptr == NULL:
                p = seg_end_ptr + 1
                continue

            code_int = _fast_atol(p, kv_ptr)
            if code_int < 0 or code_int >= 65536 or not lut_valid[code_int]:
                p = seg_end_ptr + 1
                continue

            length_i = lut_length[code_int]
            is_float_i = lut_is_float[code_int]

            val_start = kv_ptr + 1
            val_end = seg_end_ptr
            if val_start >= val_end:
                p = seg_end_ptr + 1
                continue

            # Strip brackets if present.
            if val_start[0] == OPEN_BRACKET:
                val_start += 1
            if val_end > val_start and (val_end - 1)[0] == CLOSE_BRACKET:
                val_end -= 1

            if val_start >= val_end:
                p = seg_end_ptr + 1
                continue

            if length_i == 1:
                if is_float_i:
                    float_val = _fast_atof(val_start, val_end)
                    if float_val != -3.0 or (val_end - val_start == 4 and val_start[0] == MINUS):
                        sf_ptrs[code_int][row_idx] = <float>float_val
                        scalar_valid_ptrs[code_int][row_idx] = 1
                else:
                    int_val = _fast_atol(val_start, val_end)
                    if int_val != -3 or (val_end - val_start == 2 and val_start[0] == MINUS):
                        si_ptrs[code_int][row_idx] = int_val
                        scalar_valid_ptrs[code_int][row_idx] = 1
            else:
                if is_float_i:
                    new_count = _parse_vector_float_ptr(vf_ptrs[code_int] + vo_counters[code_int], val_start, val_end, length_i)
                    vo_counters[code_int] += new_count
                else:
                    new_count = _parse_vector_int_ptr(vi_ptrs[code_int] + vo_counters[code_int], val_start, val_end, length_i)
                    vo_counters[code_int] += new_count

            p = seg_end_ptr + 1

        for i in range(num_vector_codes):
            code_iter = vector_codes[i]
            vo_ptrs[code_iter][row_idx + 1] = vo_counters[code_iter]

    return feature_buffers, meta_ints, batch_size


def parse_lines_fast(
    list lines,
    dict code_to_base_meta,
    tuple direct_columns,
    int required_col_count,
    int payload_col_idx,
    tuple metadata_columns,
    dict metadata_maps,
    object feature_buffers,
    dict meta_ints,
):
    """Legacy entry point — delegates to parse_arrow_direct when possible.

    Kept for backward-compat with callers that pass a Python list of strings.
    """
    cdef Py_ssize_t batch_size = len(lines)
    cdef Py_ssize_t row_idx, n_bytes
    cdef const char* buf
    cdef const char* buf_end
    cdef const char* col_starts[64]
    cdef const char* col_ends[64]
    cdef const char* p
    cdef const char* seg_start
    cdef const char* seg_end
    cdef const char* seg_end_ptr
    cdef const char* kv_ptr
    cdef const char* val_start
    cdef const char* val_end
    cdef int col_count
    cdef long code_int
    cdef int length_i
    cdef bint is_float_i
    cdef double float_val
    cdef long int_val
    cdef Py_ssize_t metadata_off, metadata_len
    cdef object metadata_name, metadata_value, metadata_map

    # LUTs for fast code_int resolution
    cdef bint lut_valid[65536]
    cdef int lut_length[65536]
    cdef bint lut_is_float[65536]
    cdef float* sf_ptrs[65536]
    cdef long* si_ptrs[65536]
    cdef float* vf_ptrs[65536]
    cdef long* vi_ptrs[65536]
    cdef int* vo_ptrs[65536]
    cdef uint8_t* scalar_valid_ptrs[65536]
    cdef int vo_counters[65536]

    cdef long vector_codes[2048]
    cdef int num_vector_codes = 0
    
    cdef int i
    for i in range(65536):
        lut_valid[i] = False
        vo_counters[i] = 0

    cdef long code_iter
    cdef int len_iter
    cdef bint is_float_iter

    # ID columns are scalar int64 base features; meta_ints are int32.
    cdef object scalar_validity = getattr(feature_buffers, "scalar_validity")
    cdef list direct_names = [item[0] for item in direct_columns]
    cdef list direct_indices = [item[1] for item in direct_columns]
    cdef list direct_buffers = [feature_buffers[name] for name in direct_names]
    cdef list direct_validity = [scalar_validity[name] for name in direct_names]

    cdef long[::1] si_view
    cdef float[::1] sf_view
    cdef long[::1] vi_view
    cdef float[::1] vf_view
    cdef int[::1] vo_view
    cdef uint8_t[::1] scalar_valid_view

    for code_iter, meta_tuple in code_to_base_meta.items():
        if code_iter < 0 or code_iter >= 65536:
            continue
        feat_name = meta_tuple[0]
        if feat_name in direct_names:
            continue
        len_iter = meta_tuple[1]
        is_float_iter = meta_tuple[2]
        buf_obj = feature_buffers[feat_name]
        
        lut_valid[code_iter] = True
        lut_length[code_iter] = len_iter
        lut_is_float[code_iter] = is_float_iter
        
        if len_iter == 1:
            if is_float_iter:
                sf_view = buf_obj
                sf_ptrs[code_iter] = &sf_view[0]
            else:
                si_view = buf_obj
                si_ptrs[code_iter] = &si_view[0]
            scalar_valid_view = scalar_validity[feat_name]
            scalar_valid_ptrs[code_iter] = &scalar_valid_view[0]
        else:
            flat_buf, off_buf = buf_obj
            if is_float_iter:
                vf_view = flat_buf
                vf_ptrs[code_iter] = &vf_view[0]
            else:
                vi_view = flat_buf
                vi_ptrs[code_iter] = &vi_view[0]
            vo_view = off_buf
            vo_ptrs[code_iter] = &vo_view[0]
            
            vector_codes[num_vector_codes] = code_iter
            num_vector_codes += 1

    for row_idx in range(batch_size):
        line = lines[row_idx]
        # Zero-copy access to the str's internal utf-8 buffer (CPython 3.7+
        # caches this; subsequent calls on the same str are O(1)).
        buf = PyUnicode_AsUTF8AndSize(line, &n_bytes)
        buf_end = buf + n_bytes

        if required_col_count > 64 or payload_col_idx < 0 or payload_col_idx >= required_col_count:
            raise ValueError("parser requires more than 64 source columns")
        # Split source columns by finding tabs.
        col_starts[0] = buf
        col_count = 0
        p = buf
        while p < buf_end and col_count < required_col_count - 1:
            p = <const char*>memchr(p, TAB, buf_end - p)
            if p == NULL:
                break
            col_ends[col_count] = p
            col_count += 1
            p += 1
            col_starts[col_count] = p

        if col_count != required_col_count - 1:
            raise ValueError(f"Expected {required_col_count} tab-separated columns, got {col_count + 1}")
        col_ends[required_col_count - 1] = buf_end

        for i in range(len(direct_names)):
            direct_buffers[i][row_idx] = _fast_atol(col_starts[direct_indices[i]], col_ends[direct_indices[i]])
            direct_validity[i][row_idx] = _scalar_is_present(col_starts[direct_indices[i]], col_ends[direct_indices[i]])

        for i in range(len(metadata_columns)):
            metadata_name = metadata_columns[i][0]
            metadata_off = col_starts[metadata_columns[i][1]] - buf
            metadata_len = col_ends[metadata_columns[i][1]] - col_starts[metadata_columns[i][1]]
            metadata_value = PyUnicode_FromStringAndSize(buf + metadata_off, metadata_len)
            metadata_map = metadata_maps[metadata_name]
            try:
                meta_ints[metadata_name][row_idx] = metadata_map[metadata_value]
            except KeyError as error:
                raise KeyError(f"unknown metadata value for {metadata_name}: {metadata_value!r}") from error

        # Parse feature column (col 7) -- the hot inner loop.
        seg_start = col_starts[payload_col_idx]
        seg_end = col_ends[payload_col_idx]
        if seg_start >= seg_end:
            continue

        p = seg_start
        while p < seg_end:
            seg_end_ptr = <const char*>memchr(p, MAP_SEP, seg_end - p)
            if seg_end_ptr == NULL:
                seg_end_ptr = seg_end

            if seg_end_ptr == p:
                p = seg_end_ptr + 1
                continue

            kv_ptr = <const char*>memchr(p, KV_SEP, seg_end_ptr - p)
            if kv_ptr == NULL:
                p = seg_end_ptr + 1
                continue

            code_int = _fast_atol(p, kv_ptr)
            if code_int < 0 or code_int >= 65536 or not lut_valid[code_int]:
                p = seg_end_ptr + 1
                continue

            length_i = lut_length[code_int]
            is_float_i = lut_is_float[code_int]

            val_start = kv_ptr + 1
            val_end = seg_end_ptr
            if val_start >= val_end:
                p = seg_end_ptr + 1
                continue

            # Strip brackets if present.
            if val_start[0] == OPEN_BRACKET:
                val_start += 1
            if val_end > val_start and (val_end - 1)[0] == CLOSE_BRACKET:
                val_end -= 1

            if val_start >= val_end:
                p = seg_end_ptr + 1
                continue

            if length_i == 1:
                # Scalar path -- skip "-3" sentinel writes (buffer already padded).
                if is_float_i:
                    float_val = _fast_atof(val_start, val_end)
                    if float_val != -3.0 or (val_end - val_start == 4 and val_start[0] == MINUS):
                        sf_ptrs[code_int][row_idx] = <float>float_val
                        scalar_valid_ptrs[code_int][row_idx] = 1
                else:
                    int_val = _fast_atol(val_start, val_end)
                    if int_val != -3 or (val_end - val_start == 2 and val_start[0] == MINUS):
                        si_ptrs[code_int][row_idx] = int_val
                        scalar_valid_ptrs[code_int][row_idx] = 1
            else:
                # Vector path -- _parse_vector_* writes into the prefilled buffer.
                if is_float_i:
                    new_count = _parse_vector_float_ptr(vf_ptrs[code_int] + vo_counters[code_int], val_start, val_end, length_i)
                    vo_counters[code_int] += new_count
                else:
                    new_count = _parse_vector_int_ptr(vi_ptrs[code_int] + vo_counters[code_int], val_start, val_end, length_i)
                    vo_counters[code_int] += new_count

            p = seg_end_ptr + 1

        for i in range(num_vector_codes):
            code_iter = vector_codes[i]
            vo_ptrs[code_iter][row_idx + 1] = vo_counters[code_iter]

    return feature_buffers, meta_ints, batch_size
