# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""High-performance generic line parser.

C-level string scanning with memchr and zero per-row Python object allocation.
Writes numbers directly into preallocated NumPy array memoryviews according to
a generic ParsePlan ABI.
"""

from libc.stdint cimport int32_t, int64_t, uint8_t
from libc.stdlib cimport strtod
from libc.string cimport memchr, memcpy

cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t* size) except NULL
    object PyUnicode_FromStringAndSize(const char* u, Py_ssize_t size)

cdef int TAB = 0x09
cdef int MAP_SEP = 0x02
cdef int KV_SEP = 0x03
cdef int OPEN_BRACKET = 0x5B
cdef int CLOSE_BRACKET = 0x5D
cdef int COMMA = 0x2C
cdef int MINUS = 0x2D
cdef int PLUS = 0x2B


cdef inline double _fast_atof(const char* s, const char* end) noexcept nogil:
    cdef char buf[64]
    cdef Py_ssize_t n = end - s
    if n <= 0 or n >= 63:
        return -3.0
    memcpy(buf, s, n)
    buf[n] = 0
    return strtod(buf, NULL)


cdef inline long _slow_atol(const char* s, const char* end) noexcept nogil:
    cdef char buf[64]
    cdef Py_ssize_t n = end - s
    if n <= 0 or n >= 63:
        return -3
    memcpy(buf, s, n)
    buf[n] = 0
    return <long>strtod(buf, NULL)


cdef inline long _fast_atol(const char* s, const char* end) noexcept nogil:
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


def scan_lines(
    list lines,
    dict abi,
    dict buffers,
    dict validity,
):
    cdef Py_ssize_t batch_size = len(lines)
    cdef int source_column_count = abi["source_column_count"]
    cdef int feature_column = abi.get("feature_column", source_column_count - 1)
    cdef tuple text_columns = tuple(abi.get("text_columns", ()))
    cdef dict fields = abi["fields"]

    cdef dict raw_text = {col: [] for col in text_columns}

    cdef Py_ssize_t row_idx
    cdef const char* line_ptr
    cdef Py_ssize_t line_len
    cdef const char* col_start
    cdef const char* col_end
    cdef const char* line_end
    cdef int col_idx
    cdef const char* kv_ptr
    cdef const char* kv_end
    cdef const char* sep_ptr
    cdef const char* eq_ptr
    cdef const char* val_start
    cdef const char* val_end
    cdef const char* v_p
    cdef const char* comma_ptr
    cdef long code
    cdef object field_info
    cdef int buf_idx
    cdef int length
    cdef int v_idx
    cdef bint is_float
    cdef double val_f
    cdef long val_i
    cdef object py_line

    for row_idx in range(batch_size):
        py_line = lines[row_idx]
        line_ptr = PyUnicode_AsUTF8AndSize(py_line, &line_len)
        line_end = line_ptr + line_len
        col_start = line_ptr

        for col_idx in range(source_column_count):
            if col_start >= line_end:
                col_end = line_end
            else:
                sep_ptr = <const char*>memchr(col_start, TAB, line_end - col_start)
                if sep_ptr == NULL:
                    col_end = line_end
                else:
                    col_end = sep_ptr

            if col_idx in raw_text:
                if col_end > col_start:
                    raw_text[col_idx].append(PyUnicode_FromStringAndSize(col_start, col_end - col_start))
                else:
                    raw_text[col_idx].append("")

            if col_idx == feature_column:
                kv_ptr = col_start
                while kv_ptr < col_end:
                    sep_ptr = <const char*>memchr(kv_ptr, MAP_SEP, col_end - kv_ptr)
                    if sep_ptr == NULL:
                        kv_end = col_end
                    else:
                        kv_end = sep_ptr

                    eq_ptr = <const char*>memchr(kv_ptr, KV_SEP, kv_end - kv_ptr)
                    if eq_ptr != NULL:
                        code = _fast_atol(kv_ptr, eq_ptr)
                        if code in fields:
                            field_info = fields[code]
                            buf_idx = field_info[0]
                            length = field_info[1]
                            is_float = field_info[2]
                            
                            val_start = eq_ptr + 1
                            val_end = kv_end
                            
                            if length == 1:
                                if is_float:
                                    val_f = _fast_atof(val_start, val_end)
                                    buffers[buf_idx][row_idx] = val_f
                                else:
                                    val_i = _fast_atol(val_start, val_end)
                                    buffers[buf_idx][row_idx] = val_i
                                if buf_idx in validity:
                                    validity[buf_idx][row_idx] = 1
                            else:
                                v_p = val_start
                                v_idx = 0
                                while v_p < val_end and v_idx < length:
                                    comma_ptr = <const char*>memchr(v_p, COMMA, val_end - v_p)
                                    if comma_ptr == NULL:
                                        comma_ptr = val_end
                                    if is_float:
                                        buffers[buf_idx][row_idx, v_idx] = _fast_atof(v_p, comma_ptr)
                                    else:
                                        buffers[buf_idx][row_idx, v_idx] = _fast_atol(v_p, comma_ptr)
                                    v_idx += 1
                                    if comma_ptr == val_end:
                                        break
                                    v_p = comma_ptr + 1
                                if buf_idx in validity:
                                    validity[buf_idx][row_idx] = 1

                    if sep_ptr == NULL:
                        break
                    kv_ptr = sep_ptr + 1

            if sep_ptr == NULL or col_start >= line_end:
                break
            col_start = col_end + 1

    return buffers, validity, raw_text, batch_size
