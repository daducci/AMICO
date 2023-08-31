# distutils: language = c++
# cython: language_level = 3

cdef int dir_to_lut_idx(double [::1]direction, short [::1]hash_table) nogil
