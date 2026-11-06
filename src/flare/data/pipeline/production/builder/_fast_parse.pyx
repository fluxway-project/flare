# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated line parsing.

C-level string scanning (memchr) + number conversion (strtod/strtol) + exact lexical category lookup.
Unified single-pass parsing: writes directly into preallocated numpy buffers via typed
memoryviews -- zero per-row PyObject allocation, zero per-line bytes copy.

API:
    parse_arrow_direct(data_buf, offsets, n_rows, code_to_base_meta,
                       direct_columns, required_col_count, payload_col_idx,
                       metadata_columns, metadata_table, feature_buffers,
                       meta_ints, category_table, null_sentinel, float_null_sentinel)
    -> (feature_buffers, meta_ints, batch_size)
"""

from cpython.bytes cimport PyBytes_AS_STRING
from libc.float cimport FLT_MAX
from libc.limits cimport LONG_MAX, LONG_MIN
from libc.math cimport isfinite
from libc.stdint cimport int32_t, uint8_t, uint64_t
from libc.stdlib cimport calloc, free, malloc, strtod
from libc.string cimport memchr, memcmp, memcpy
import numpy as np

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
cdef int DOT = 0x2E            # '.'

cdef struct TokenSlot:
    const char* data
    Py_ssize_t length
    uint64_t fingerprint
    int value
    int occupied


cdef struct ParseScratch:
    bint lut_valid[65536]
    int lut_length[65536]
    bint lut_is_float[65536]
    bint lut_has_category[65536]
    bint lut_has_numeric[65536]
    float* sf_ptrs[65536]
    long* si_ptrs[65536]
    float* vf_ptrs[65536]
    long* vi_ptrs[65536]
    int* vo_ptrs[65536]
    int32_t* cat_s_ptrs[65536]
    int32_t* cat_v_ptrs[65536]
    int* cat_vo_ptrs[65536]
    uint8_t* scalar_valid_ptrs[65536]
    uint8_t* cat_valid_ptrs[65536]
    int vo_counters[65536]


cdef class _ParseScratch:
    cdef ParseScratch* data

    def __cinit__(self):
        self.data = <ParseScratch*>malloc(sizeof(ParseScratch))
        if self.data == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.data)


cdef inline uint64_t _fingerprint(const char* data, Py_ssize_t length) noexcept nogil:
    cdef uint64_t value = (<uint64_t>0xCBF29CE4 << 32) | <uint64_t>0x84222325
    cdef uint64_t prime = (<uint64_t>0x100 << 32) | <uint64_t>0x1B3
    cdef Py_ssize_t index
    for index in range(length):
        value = (value ^ <unsigned char>data[index]) * prime
    return value


cdef class CategoryTable:
    cdef TokenSlot** _slots
    cdef unsigned int* _masks
    cdef int* _default_indices
    cdef TokenSlot** _owned_slots
    cdef Py_ssize_t _owned_count
    cdef object _tokens
    cdef object _codes

    def __cinit__(self):
        self._owned_slots = NULL
        self._owned_count = 0
        self._slots = <TokenSlot**>calloc(65536, sizeof(TokenSlot*))
        self._masks = <unsigned int*>calloc(65536, sizeof(unsigned int))
        self._default_indices = <int*>calloc(65536, sizeof(int))
        if self._slots == NULL or self._masks == NULL or self._default_indices == NULL:
            raise MemoryError()
        self._tokens = []
        self._codes = frozenset()

    def __dealloc__(self):
        cdef Py_ssize_t index
        if self._owned_slots != NULL:
            for index in range(self._owned_count):
                free(self._owned_slots[index])
            free(self._owned_slots)
        if self._slots != NULL:
            free(self._slots)
        if self._masks != NULL:
            free(self._masks)
        if self._default_indices != NULL:
            free(self._default_indices)

    def __init__(self, category_mappings, default_indices):
        cdef int code, size
        cdef unsigned int position
        cdef uint64_t fingerprint
        cdef TokenSlot* slots
        cdef object token, encoded, ordinal
        cdef int default_idx
        self._codes = frozenset(category_mappings)
        if self._codes != frozenset(default_indices):
            raise ValueError("category defaults must match vocabulary codes")
        if not category_mappings:
            return
        self._owned_slots = <TokenSlot**>calloc(len(category_mappings), sizeof(TokenSlot*))
        if self._owned_slots == NULL:
            raise MemoryError()
        shared_by_size = {}
        for code, mapping in category_mappings.items():
            if type(code) is not int or code < 0 or code >= 65536 or not mapping:
                raise ValueError(f"invalid category mapping for feature code {code}")
            default_idx = default_indices[code]
            if type(default_idx) is not int or default_idx < 0 or default_idx > 2147483647:
                raise ValueError(f"invalid category default for feature code {code}")
            self._default_indices[code] = default_idx
            for owner_code, owner_mapping in shared_by_size.get(len(mapping), ()):
                if mapping == owner_mapping:
                    self._slots[code] = self._slots[owner_code]
                    self._masks[code] = self._masks[owner_code]
                    break
            else:
                size = 1
                while size < 2 * len(mapping):
                    size *= 2
                slots = <TokenSlot*>calloc(size, sizeof(TokenSlot))
                if slots == NULL:
                    raise MemoryError()
                self._owned_slots[self._owned_count] = slots
                self._owned_count += 1
                self._slots[code] = slots
                self._masks[code] = size - 1
                for token, ordinal in mapping.items():
                    if (
                        type(token) is not str or not token
                        or type(ordinal) is not int or ordinal < 0 or ordinal > 2147483647
                    ):
                        raise TypeError(f"invalid category entry for feature code {code}")
                    encoded = token.encode("utf-8")
                    self._tokens.append(encoded)
                    fingerprint = _fingerprint(PyBytes_AS_STRING(encoded), len(encoded))
                    position = <unsigned int>fingerprint & self._masks[code]
                    while slots[position].occupied:
                        position = (position + 1) & self._masks[code]
                    slots[position].data = PyBytes_AS_STRING(encoded)
                    slots[position].length = len(encoded)
                    slots[position].fingerprint = fingerprint
                    slots[position].value = ordinal
                    slots[position].occupied = 1
                shared_by_size.setdefault(len(mapping), []).append((code, mapping))

    cdef inline int lookup(self, int code, const char* data, Py_ssize_t length) noexcept nogil:
        cdef TokenSlot* slots = self._slots[code]
        cdef unsigned int mask = self._masks[code]
        cdef uint64_t fingerprint = _fingerprint(data, length)
        cdef unsigned int position = <unsigned int>fingerprint & mask
        while slots[position].occupied:
            if (
                slots[position].fingerprint == fingerprint
                and slots[position].length == length
                and memcmp(slots[position].data, data, length) == 0
            ):
                return slots[position].value
            position = (position + 1) & mask
        return self._default_indices[code]

    def has_code(self, int code):
        return code >= 0 and code < 65536 and self._slots[code] != NULL

    @property
    def codes(self):
        return self._codes


cdef class MetadataTable:
    cdef TokenSlot** _slots
    cdef unsigned int* _masks
    cdef Py_ssize_t _num_cols
    cdef object _column_names
    cdef object _tokens

    def __cinit__(self, metadata_slots, metadata_maps):
        self._num_cols = len(metadata_slots)
        self._column_names = tuple(slot[0] for slot in metadata_slots)
        self._tokens = []
        if self._num_cols == 0:
            self._slots = NULL
            self._masks = NULL
            return
        self._slots = <TokenSlot**>calloc(self._num_cols, sizeof(TokenSlot*))
        self._masks = <unsigned int*>calloc(self._num_cols, sizeof(unsigned int))
        if self._slots == NULL or self._masks == NULL:
            raise MemoryError()

        cdef int i, size
        cdef unsigned int position
        cdef uint64_t fingerprint
        cdef TokenSlot* slots
        cdef object name, token, encoded, ordinal, mapping

        for i in range(self._num_cols):
            name = metadata_slots[i][0]
            mapping = metadata_maps[name]
            size = 1
            while size < 2 * len(mapping) + 2:
                size *= 2
            slots = <TokenSlot*>calloc(size, sizeof(TokenSlot))
            if slots == NULL:
                raise MemoryError()
            self._slots[i] = slots
            self._masks[i] = size - 1
            for token, ordinal in mapping.items():
                if type(token) is not str:
                    raise TypeError(f"metadata key for {name!r} must be text")
                if type(ordinal) is not int or ordinal < 0 or ordinal > 2147483647:
                    raise ValueError(f"metadata index for {name!r} must be non-negative int32")
                encoded = token.encode("utf-8")
                self._tokens.append(encoded)
                fingerprint = _fingerprint(PyBytes_AS_STRING(encoded), len(encoded))
                position = <unsigned int>fingerprint & self._masks[i]
                while slots[position].occupied:
                    position = (position + 1) & self._masks[i]
                slots[position].data = PyBytes_AS_STRING(encoded)
                slots[position].length = len(encoded)
                slots[position].fingerprint = fingerprint
                slots[position].value = ordinal
                slots[position].occupied = 1

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self._slots != NULL:
            for i in range(self._num_cols):
                if self._slots[i] != NULL:
                    free(self._slots[i])
            free(self._slots)
        if self._masks != NULL:
            free(self._masks)

    cdef inline int lookup(self, int col_idx, const char* data, Py_ssize_t length) noexcept nogil:
        if col_idx < 0 or col_idx >= self._num_cols or self._slots[col_idx] == NULL:
            return -1
        cdef TokenSlot* slots = self._slots[col_idx]
        cdef unsigned int mask = self._masks[col_idx]
        cdef uint64_t fingerprint = _fingerprint(data, length)
        cdef unsigned int position = <unsigned int>fingerprint & mask
        while slots[position].occupied:
            if (
                slots[position].fingerprint == fingerprint
                and slots[position].length == length
                and memcmp(slots[position].data, data, length) == 0
            ):
                return slots[position].value
            position = (position + 1) & mask
        return -1


cdef inline bint _scalar_is_present(const char* start, const char* end) noexcept nogil:
    return start < end and not (end - start == 1 and start[0] == 0x30)


cdef inline bint _parse_float(const char* s, const char* end, double* out) noexcept nogil:
    cdef char buf[64]
    cdef Py_ssize_t n = end - s
    cdef char* endptr
    cdef const char* first = s
    if n <= 0 or n >= 63:
        return False
    if first[0] == MINUS or first[0] == PLUS:
        first += 1
    if first >= end or not (
        (first[0] >= 0x30 and first[0] <= 0x39) or first[0] == DOT
    ):
        return False
    memcpy(buf, s, n)
    buf[n] = 0
    out[0] = strtod(buf, &endptr)
    return endptr == buf + n and isfinite(out[0]) and -FLT_MAX <= out[0] <= FLT_MAX


cdef inline bint _float_to_long(double value, long* out) noexcept nogil:
    if <long double>value < <long double>LONG_MIN or <long double>value >= <long double>LONG_MAX + 1:
        return False
    out[0] = <long>value
    return True


cdef inline bint _parse_int(const char* s, const char* end, long* out) noexcept nogil:
    cdef const char* p = s
    cdef Py_ssize_t digits
    cdef unsigned long value = 0
    cdef unsigned long limit = <unsigned long>LONG_MAX
    cdef int digit
    cdef bint negative = False
    cdef double decimal

    if p >= end:
        return False
    if p[0] == MINUS:
        negative = True
        limit += 1
        p += 1
    elif p[0] == PLUS:
        p += 1
    digits = end - p
    if digits <= 0:
        return False
    while p < end:
        digit = p[0] - 0x30
        if digit < 0 or digit > 9:
            if not _parse_float(s, end, &decimal):
                return False
            return _float_to_long(decimal, out)
        if value > (limit - digit) // 10:
            return False
        value = value * 10 + digit
        p += 1
    if negative:
        out[0] = LONG_MIN if value == limit else -<long>value
    else:
        out[0] = <long>value
    return True


def parse_arrow_direct(
    const char[::1] data_buf,
    const long[::1] offsets,
    Py_ssize_t n_rows,
    dict code_to_base_meta,
    tuple direct_columns,
    int required_col_count,
    int payload_col_idx,
    tuple metadata_columns,
    MetadataTable metadata_table,
    object feature_buffers,
    dict meta_ints,
    CategoryTable category_table,
    long null_sentinel,
    double float_null_sentinel,
):
    """Zero-copy Arrow buffer -> parsed feature buffers (numeric + category)."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if offsets.shape[0] < n_rows + 1:
        raise ValueError("offsets buffer smaller than n_rows + 1")
    if required_col_count > 64 or payload_col_idx < 0 or payload_col_idx >= required_col_count:
        raise ValueError(f"invalid required_col_count ({required_col_count}) or payload_col_idx ({payload_col_idx})")
    if category_table is not None:
        for category_code in category_table.codes:
            if category_code not in code_to_base_meta:
                raise ValueError(f"category feature code {category_code} is not parsed")
            if f"category_index_{category_code}" not in feature_buffers:
                raise ValueError(f"category output is missing for feature code {category_code}")
    if metadata_table is None:
        raise TypeError("metadata table is required")
    if metadata_table._column_names != tuple(slot[0] for slot in metadata_columns):
        raise ValueError("metadata table columns differ from parser columns")
    if n_rows == 0:
        return feature_buffers, meta_ints, 0
    if offsets[0] < 0 or offsets[n_rows] > data_buf.shape[0]:
        raise ValueError("offsets out of bounds for data buffer")

    cdef char empty_data = 0
    cdef const char* data_ptr
    if data_buf.shape[0] == 0:
        data_ptr = &empty_data
    else:
        data_ptr = <const char*>&data_buf[0]
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
    cdef bint has_cat_i
    cdef bint has_num_i
    cdef double float_val = 0.0
    cdef long int_val = 0
    cdef int meta_val = 0

    cdef long code_iter
    cdef int len_iter
    cdef bint is_float_iter
    cdef bint has_cat, has_num

    cdef _ParseScratch scratch = _ParseScratch()
    cdef bint* lut_valid = scratch.data.lut_valid
    cdef int* lut_length = scratch.data.lut_length
    cdef bint* lut_is_float = scratch.data.lut_is_float
    cdef bint* lut_has_category = scratch.data.lut_has_category
    cdef bint* lut_has_numeric = scratch.data.lut_has_numeric
    cdef float** sf_ptrs = scratch.data.sf_ptrs
    cdef long** si_ptrs = scratch.data.si_ptrs
    cdef float** vf_ptrs = scratch.data.vf_ptrs
    cdef long** vi_ptrs = scratch.data.vi_ptrs
    cdef int** vo_ptrs = scratch.data.vo_ptrs
    cdef int32_t** cat_s_ptrs = scratch.data.cat_s_ptrs
    cdef int32_t** cat_v_ptrs = scratch.data.cat_v_ptrs
    cdef int** cat_vo_ptrs = scratch.data.cat_vo_ptrs
    cdef uint8_t** scalar_valid_ptrs = scratch.data.scalar_valid_ptrs
    cdef uint8_t** cat_valid_ptrs = scratch.data.cat_valid_ptrs
    cdef int* vo_counters = scratch.data.vo_counters

    cdef int[::1] vector_codes = np.empty(len(code_to_base_meta), dtype=np.int32)
    cdef int num_vector_codes = 0

    cdef int i
    for i in range(65536):
        lut_valid[i] = False
        vo_counters[i] = 0

    cdef object scalar_validity = getattr(feature_buffers, "scalar_validity")
    cdef int num_direct = len(direct_columns)
    cdef long* direct_ptrs[64]
    cdef uint8_t* direct_valid_ptrs[64]
    cdef int direct_col_indices[64]
    cdef long[::1] d_view
    cdef uint8_t[::1] dv_view
    cdef object direct_name

    for i in range(num_direct):
        direct_name = direct_columns[i][0]
        direct_col_indices[i] = direct_columns[i][1]
        d_view = feature_buffers[direct_name]
        dv_view = scalar_validity[direct_name]
        if d_view.shape[0] < batch_size or dv_view.shape[0] < batch_size:
            raise ValueError("direct feature buffer is smaller than the batch")
        direct_ptrs[i] = &d_view[0]
        direct_valid_ptrs[i] = &dv_view[0]

    cdef int num_meta = len(metadata_columns)
    cdef int32_t* meta_ptrs[64]
    cdef int meta_col_indices[64]
    cdef int32_t[::1] m_view
    cdef object meta_name

    for i in range(num_meta):
        meta_name = metadata_columns[i][0]
        meta_col_indices[i] = metadata_columns[i][1]
        m_view = meta_ints[meta_name]
        if m_view.shape[0] < batch_size:
            raise ValueError(f"metadata buffer is smaller than the batch for {meta_name!r}")
        meta_ptrs[i] = &m_view[0]

    cdef long[::1] si_view
    cdef float[::1] sf_view
    cdef long[::1] vi_view
    cdef float[::1] vf_view
    cdef int[::1] vo_view
    cdef int32_t[::1] cat_s_view
    cdef int32_t[::1] cat_v_view
    cdef int[::1] cat_vo_view
    cdef uint8_t[::1] scalar_valid_view
    cdef uint8_t[::1] cat_valid_view
    cdef set direct_names = {item[0] for item in direct_columns}

    for code_iter, meta_tuple in code_to_base_meta.items():
        if code_iter < 0 or code_iter >= 65536:
            raise ValueError(f"invalid feature code {code_iter}")
        feat_name = meta_tuple[0]
        if feat_name in direct_names:
            continue
        len_iter = meta_tuple[1]
        is_float_iter = meta_tuple[2]
        if len_iter < 1:
            raise ValueError(f"invalid feature length for code {code_iter}")
        cat_col_name = f"category_index_{code_iter}"

        has_cat = category_table is not None and category_table.has_code(code_iter)
        has_num = (feat_name in feature_buffers)
        if not has_cat and not has_num:
            continue

        lut_valid[code_iter] = True
        lut_length[code_iter] = len_iter
        lut_is_float[code_iter] = is_float_iter
        lut_has_category[code_iter] = has_cat
        lut_has_numeric[code_iter] = has_num
        if len_iter > 1:
            vector_codes[num_vector_codes] = code_iter
            num_vector_codes += 1

        if has_num:
            buf_obj = feature_buffers[feat_name]
            if len_iter == 1:
                if is_float_iter:
                    sf_view = buf_obj
                    if sf_view.shape[0] < batch_size:
                        raise ValueError(f"numeric scalar buffer is too small for code {code_iter}")
                    sf_ptrs[code_iter] = &sf_view[0]
                else:
                    si_view = buf_obj
                    if si_view.shape[0] < batch_size:
                        raise ValueError(f"numeric scalar buffer is too small for code {code_iter}")
                    si_ptrs[code_iter] = &si_view[0]
                scalar_valid_view = scalar_validity[feat_name]
                if scalar_valid_view.shape[0] < batch_size:
                    raise ValueError(f"numeric validity buffer is too small for code {code_iter}")
                scalar_valid_ptrs[code_iter] = &scalar_valid_view[0]
            else:
                flat_buf, off_buf = buf_obj
                if is_float_iter:
                    vf_view = flat_buf
                    if vf_view.shape[0] // len_iter < batch_size:
                        raise ValueError(f"numeric vector buffer is too small for code {code_iter}")
                    vf_ptrs[code_iter] = &vf_view[0]
                else:
                    vi_view = flat_buf
                    if vi_view.shape[0] // len_iter < batch_size:
                        raise ValueError(f"numeric vector buffer is too small for code {code_iter}")
                    vi_ptrs[code_iter] = &vi_view[0]
                vo_view = off_buf
                if vo_view.shape[0] < batch_size + 1:
                    raise ValueError(f"numeric offset buffer is too small for code {code_iter}")
                vo_ptrs[code_iter] = &vo_view[0]
        else:
            sf_ptrs[code_iter] = NULL
            si_ptrs[code_iter] = NULL
            vf_ptrs[code_iter] = NULL
            vi_ptrs[code_iter] = NULL
            vo_ptrs[code_iter] = NULL
            scalar_valid_ptrs[code_iter] = NULL

        if has_cat:
            cat_buf_obj = feature_buffers[cat_col_name]
            if len_iter == 1:
                cat_s_view = cat_buf_obj
                if cat_s_view.shape[0] < batch_size:
                    raise ValueError(f"category scalar buffer is too small for code {code_iter}")
                cat_s_ptrs[code_iter] = &cat_s_view[0]
                cat_valid_view = scalar_validity[cat_col_name]
                if cat_valid_view.shape[0] < batch_size:
                    raise ValueError(f"category validity buffer is too small for code {code_iter}")
                cat_valid_ptrs[code_iter] = &cat_valid_view[0]
            else:
                flat_cat, off_cat = cat_buf_obj
                cat_v_view = flat_cat
                if cat_v_view.shape[0] // len_iter < batch_size:
                    raise ValueError(f"category vector buffer is too small for code {code_iter}")
                cat_v_ptrs[code_iter] = &cat_v_view[0]
                cat_vo_view = off_cat
                if cat_vo_view.shape[0] < batch_size + 1:
                    raise ValueError(f"category offset buffer is too small for code {code_iter}")
                cat_vo_ptrs[code_iter] = &cat_vo_view[0]
        else:
            cat_s_ptrs[code_iter] = NULL
            cat_v_ptrs[code_iter] = NULL
            cat_vo_ptrs[code_iter] = NULL
            cat_valid_ptrs[code_iter] = NULL

    cdef const int* vec_codes_ptr = &vector_codes[0] if num_vector_codes > 0 else NULL
    cdef int error_code = 0
    cdef int error_arg = 0
    cdef Py_ssize_t error_row_idx = 0
    cdef const char* token_p
    cdef const char* comma_ptr
    cdef const char* token_end
    cdef int token_len, vector_base
    cdef bint retain_token
    cdef const char* col_s
    cdef const char* col_e
    cdef const char* m_s
    cdef const char* m_e

    with nogil:
        for row_idx in range(batch_size):
            if (
                off_ptr[row_idx] < 0
                or off_ptr[row_idx + 1] < off_ptr[row_idx]
                or off_ptr[row_idx + 1] > data_buf.shape[0]
            ):
                error_code = 1
                error_arg = row_idx
                break
            buf = data_ptr + off_ptr[row_idx]
            n_bytes = off_ptr[row_idx + 1] - off_ptr[row_idx]
            buf_end = buf + n_bytes

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
                error_code = 2
                error_arg = col_count + 1
                error_row_idx = row_idx
                break
            col_ends[required_col_count - 1] = buf_end

            for i in range(num_direct):
                col_s = col_starts[direct_col_indices[i]]
                col_e = col_ends[direct_col_indices[i]]
                if col_s == col_e:
                    direct_ptrs[i][row_idx] = null_sentinel
                    direct_valid_ptrs[i][row_idx] = 0
                elif _parse_int(col_s, col_e, &int_val):
                    direct_ptrs[i][row_idx] = int_val
                    direct_valid_ptrs[i][row_idx] = 1 if _scalar_is_present(col_s, col_e) else 0
                else:
                    error_code = 6
                    error_arg = direct_col_indices[i]
                    error_row_idx = row_idx
                    break
            if error_code != 0:
                break

            for i in range(num_meta):
                m_s = col_starts[meta_col_indices[i]]
                m_e = col_ends[meta_col_indices[i]]
                meta_val = metadata_table.lookup(i, m_s, m_e - m_s)
                if meta_val < 0:
                    error_code = 3
                    error_arg = i
                    error_row_idx = row_idx
                    break
                meta_ptrs[i][row_idx] = meta_val
            if error_code != 0:
                break

            # Parse feature payload column.
            seg_start = col_starts[payload_col_idx]
            seg_end = col_ends[payload_col_idx]
            if payload_col_idx == required_col_count - 1:
                seg_end_ptr = <const char*>memchr(seg_start, TAB, seg_end - seg_start)
                if seg_end_ptr != NULL:
                    seg_end = seg_end_ptr
            if seg_start >= seg_end:
                for i in range(num_vector_codes):
                    code_iter = vec_codes_ptr[i]
                    if vo_ptrs[code_iter] != NULL:
                        vo_ptrs[code_iter][row_idx + 1] = vo_counters[code_iter]
                    if cat_vo_ptrs[code_iter] != NULL and cat_vo_ptrs[code_iter] != vo_ptrs[code_iter]:
                        cat_vo_ptrs[code_iter][row_idx + 1] = vo_counters[code_iter]
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

                if not _parse_int(p, kv_ptr, &code_int):
                    p = seg_end_ptr + 1
                    continue
                if code_int < 0 or code_int >= 65536 or not lut_valid[code_int]:
                    p = seg_end_ptr + 1
                    continue

                length_i = lut_length[code_int]
                is_float_i = lut_is_float[code_int]
                has_cat_i = lut_has_category[code_int]
                has_num_i = lut_has_numeric[code_int]

                val_start = kv_ptr + 1
                val_end = seg_end_ptr
                if val_start < val_end and val_start[0] == OPEN_BRACKET:
                    val_start += 1
                if val_end > val_start and (val_end - 1)[0] == CLOSE_BRACKET:
                    val_end -= 1

                if val_start >= val_end:
                    p = seg_end_ptr + 1
                    continue

                if length_i == 1:
                    if has_cat_i and category_table is not None:
                        cat_s_ptrs[code_int][row_idx] = category_table.lookup(code_int, val_start, val_end - val_start)
                        if cat_valid_ptrs[code_int] != NULL:
                            cat_valid_ptrs[code_int][row_idx] = 1
                    if has_num_i:
                        if is_float_i:
                            if not _parse_float(val_start, val_end, &float_val):
                                error_code = 5
                                error_arg = code_int
                                error_row_idx = row_idx
                                break
                            sf_ptrs[code_int][row_idx] = <float>float_val
                            scalar_valid_ptrs[code_int][row_idx] = 1
                        else:
                            if memchr(val_start, DOT, val_end - val_start) != NULL:
                                if not _parse_float(val_start, val_end, &float_val) or not _float_to_long(float_val, &int_val):
                                    error_code = 5
                                    error_arg = code_int
                                    error_row_idx = row_idx
                                    break
                                if float_val != float_null_sentinel:
                                    si_ptrs[code_int][row_idx] = int_val
                                    scalar_valid_ptrs[code_int][row_idx] = 1
                            else:
                                if not _parse_int(val_start, val_end, &int_val):
                                    error_code = 5
                                    error_arg = code_int
                                    error_row_idx = row_idx
                                    break
                                si_ptrs[code_int][row_idx] = int_val
                                scalar_valid_ptrs[code_int][row_idx] = 1
                else:
                    token_p = val_start
                    if has_num_i:
                        vector_base = vo_ptrs[code_int][row_idx]
                    else:
                        vector_base = cat_vo_ptrs[code_int][row_idx]
                    while token_p < val_end:
                        comma_ptr = <const char*>memchr(token_p, COMMA, val_end - token_p)
                        if comma_ptr == NULL:
                            token_end = val_end
                        else:
                            token_end = comma_ptr
                        token_len = token_end - token_p
                        if token_len > 0:
                            retain_token = has_cat_i
                            if has_num_i:
                                if is_float_i:
                                    if not _parse_float(token_p, token_end, &float_val):
                                        error_code = 5
                                        error_arg = code_int
                                        error_row_idx = row_idx
                                        break
                                    retain_token = True
                                else:
                                    if memchr(token_p, DOT, token_len) != NULL:
                                        if not _parse_float(token_p, token_end, &float_val) or not _float_to_long(float_val, &int_val):
                                            error_code = 5
                                            error_arg = code_int
                                            error_row_idx = row_idx
                                            break
                                        if has_cat_i or float_val != float_null_sentinel:
                                            retain_token = True
                                    else:
                                        if not _parse_int(token_p, token_end, &int_val):
                                            error_code = 5
                                            error_arg = code_int
                                            error_row_idx = row_idx
                                            break
                                        retain_token = True
                            if retain_token:
                                if vo_counters[code_int] - vector_base >= length_i:
                                    error_code = 4
                                    error_arg = code_int
                                    break
                                if has_cat_i:
                                    cat_v_ptrs[code_int][vo_counters[code_int]] = category_table.lookup(code_int, token_p, token_len)
                                if has_num_i:
                                    if is_float_i:
                                        vf_ptrs[code_int][vo_counters[code_int]] = <float>float_val
                                    else:
                                        vi_ptrs[code_int][vo_counters[code_int]] = int_val
                                vo_counters[code_int] += 1
                        if comma_ptr == NULL:
                            break
                        token_p = comma_ptr + 1

                if error_code != 0:
                    break
                p = seg_end_ptr + 1

            if error_code != 0:
                break

            for i in range(num_vector_codes):
                code_iter = vec_codes_ptr[i]
                if vo_ptrs[code_iter] != NULL:
                    vo_ptrs[code_iter][row_idx + 1] = vo_counters[code_iter]
                if cat_vo_ptrs[code_iter] != NULL and cat_vo_ptrs[code_iter] != vo_ptrs[code_iter]:
                    cat_vo_ptrs[code_iter][row_idx + 1] = vo_counters[code_iter]

    if error_code == 1:
        raise ValueError(f"invalid Arrow row offsets at row {error_arg}")
    elif error_code == 2:
        raise ValueError(f"Expected {required_col_count} tab-separated columns, got {error_arg}")
    elif error_code == 3:
        meta_name = metadata_columns[error_arg][0]
        meta_col = metadata_columns[error_arg][1]
        row_bytes = bytes(data_buf[off_ptr[error_row_idx]:off_ptr[error_row_idx + 1]])
        parts = row_bytes.split(b"\t")
        unknown_val = parts[meta_col].decode("utf-8", errors="replace") if meta_col < len(parts) else ""
        raise KeyError(f"unknown metadata value for {meta_name}: {unknown_val!r}")
    elif error_code == 4:
        raise ValueError(f"category vector exceeds configured length for code {error_arg}")
    elif error_code == 5:
        raise ValueError(f"invalid numeric feature value for code {error_arg} at row {error_row_idx}")
    elif error_code == 6:
        raise ValueError(f"invalid numeric direct value at column {error_arg} row {error_row_idx}")

    return feature_buffers, meta_ints, batch_size
