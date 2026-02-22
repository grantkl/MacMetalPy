/*
 * _accelerator.c — CPython C extension for macmetalpy hot-path acceleration.
 *
 * Provides legacy functions (backward compatible):
 *   wrap_result(ndarray_type, np_array)           — fast ndarray construction
 *   binary_op_fast(a, b, np_func, threshold)      — fused binary op
 *   unary_op_fast(a, np_func, threshold)          — fused unary op
 *
 * New dispatch-table functions (METH_FASTCALL):
 *   init_dispatch(ndarray_type, bool_dtype, binary_list, unary_list, cmp_list)
 *   fast_binary(a, b, op_id)
 *   fast_unary(a, op_id)
 *   fast_cmp(a, b, op_id)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* ------------------------------------------------------------------ */
/* Pre-interned attribute name strings (initialised in PyInit)        */
/* These are the PRIVATE attribute names on macmetalpy.ndarray        */
/* ------------------------------------------------------------------ */
static PyObject *str_buffer   = NULL;   /* "_buffer"  */
static PyObject *str_np_data  = NULL;   /* "_np_data" */
static PyObject *str_shape    = NULL;   /* "_shape"   */
static PyObject *str_dtype    = NULL;   /* "_dtype"   */
static PyObject *str_strides  = NULL;   /* "_strides" */
static PyObject *str_offset   = NULL;   /* "_offset"  */
static PyObject *str_base     = NULL;   /* "_base"    */

/* Interned string for dtype.kind check */
static PyObject *str_kind     = NULL;   /* "kind"     */

/* Cached empty tuple used by tp_new calls (legacy path) */
static PyObject *empty_tuple  = NULL;

/* Cached integer 0 for _offset */
static PyObject *int_zero     = NULL;

/* ------------------------------------------------------------------ */
/* Dispatch table infrastructure                                      */
/* ------------------------------------------------------------------ */
#define MAX_OPS 64
#define FLAG_FLOAT_ONLY 1

typedef struct {
    PyObject   *np_func;    /* numpy function to call (e.g. np.add) */
    Py_ssize_t  threshold;  /* size threshold for fast path */
    int         flags;      /* FLAG_FLOAT_ONLY etc. */
} OpEntry;

static OpEntry binary_table[MAX_OPS];
static OpEntry unary_table[MAX_OPS];
static OpEntry cmp_table[MAX_OPS];

static PyTypeObject *cached_ndarray_type = NULL;
static PyObject     *cached_bool_dtype   = NULL;
static PyObject     *cached_float64_dtype = NULL;

/* ------------------------------------------------------------------ */
/* Direct dict access — version-guarded for 3.12+ managed dicts       */
/* ------------------------------------------------------------------ */
static inline PyObject *
get_dict(PyObject *obj)
{
#if PY_VERSION_HEX >= 0x030c0000
    /* 3.12+ uses managed dicts; GenericGetDict returns a NEW reference */
    return PyObject_GenericGetDict(obj, NULL);
#else
    /* Pre-3.12: _PyObject_GetDictPtr returns pointer to borrowed ref */
    PyObject **dictptr = _PyObject_GetDictPtr(obj);
    if (dictptr == NULL || *dictptr == NULL)
        return NULL;
    Py_INCREF(*dictptr);   /* return a new reference for uniform handling */
    return *dictptr;
#endif
}

/* Read an attribute from an object's instance dict (returns borrowed ref) */
static inline PyObject *
dict_read(PyObject *obj, PyObject *key)
{
    PyObject *dict = get_dict(obj);
    if (!dict)
        return NULL;
    PyObject *val = PyDict_GetItem(dict, key);  /* borrowed ref */
    Py_DECREF(dict);
    return val;
}

/* Ensure an object has a __dict__ and return it (new reference) */
static inline PyObject *
ensure_dict(PyObject *obj)
{
#if PY_VERSION_HEX >= 0x030c0000
    /* 3.12+: GenericGetDict creates the dict if needed, returns new ref */
    return PyObject_GenericGetDict(obj, NULL);
#else
    PyObject **dictptr = _PyObject_GetDictPtr(obj);
    if (dictptr == NULL) return NULL;
    if (*dictptr == NULL) {
        *dictptr = PyDict_New();
        if (*dictptr == NULL) return NULL;
    }
    Py_INCREF(*dictptr);
    return *dictptr;
#endif
}

/* Write an attribute to an object's instance dict */
static inline int
dict_write(PyObject *obj, PyObject *key, PyObject *val)
{
    PyObject *dict = ensure_dict(obj);
    if (!dict) {
        PyErr_SetString(PyExc_RuntimeError, "object has no __dict__");
        return -1;
    }
    int rc = PyDict_SetItem(dict, key, val);
    Py_DECREF(dict);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Helper: build a Python tuple of C-contiguous element strides       */
/*         from a numpy array's shape (NOT from a Python tuple)       */
/* ------------------------------------------------------------------ */
static PyObject *
compute_c_strides_from_ndarray(PyArrayObject *arr)
{
    int ndim = PyArray_NDIM(arr);
    npy_intp *dims = PyArray_DIMS(arr);
    PyObject *strides = PyTuple_New(ndim);
    if (!strides) return NULL;

    if (ndim > 0) {
        Py_ssize_t stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            PyObject *v = PyLong_FromSsize_t(stride);
            if (!v) { Py_DECREF(strides); return NULL; }
            PyTuple_SET_ITEM(strides, i, v);  /* steals ref */
            stride *= (Py_ssize_t)dims[i];
        }
    }
    return strides;
}

/* ------------------------------------------------------------------ */
/* Helper: build a Python tuple from a numpy array's shape            */
/* ------------------------------------------------------------------ */
static PyObject *
shape_tuple_from_ndarray(PyArrayObject *arr)
{
    int ndim = PyArray_NDIM(arr);
    npy_intp *dims = PyArray_DIMS(arr);
    PyObject *shape = PyTuple_New(ndim);
    if (!shape) return NULL;

    for (int i = 0; i < ndim; i++) {
        PyObject *v = PyLong_FromSsize_t((Py_ssize_t)dims[i]);
        if (!v) { Py_DECREF(shape); return NULL; }
        PyTuple_SET_ITEM(shape, i, v);  /* steals ref */
    }
    return shape;
}

/* ------------------------------------------------------------------ */
/* Legacy helper: create ndarray via tp_new + PyObject_SetAttr        */
/* np_result MUST be a numpy ndarray                                  */
/* ------------------------------------------------------------------ */
static PyObject *
wrap_result_impl(PyTypeObject *tp, PyObject *np_result)
{
    PyArrayObject *arr = (PyArrayObject *)np_result;

    /* Allocate via tp_new (equivalent to cls.__new__(cls)) — skips __init__ */
    PyObject *obj = tp->tp_new(tp, empty_tuple, NULL);
    if (!obj) return NULL;

    /* Build shape tuple from numpy array's C shape */
    PyObject *shape = shape_tuple_from_ndarray(arr);
    if (!shape) { Py_DECREF(obj); return NULL; }

    /* Get dtype from numpy array (borrows reference, must incref) */
    PyArray_Descr *descr = PyArray_DESCR(arr);
    PyObject *dtype = (PyObject *)descr;
    Py_INCREF(dtype);

    /* Compute element strides from shape */
    PyObject *strides = compute_c_strides_from_ndarray(arr);
    if (!strides) { Py_DECREF(dtype); Py_DECREF(shape); Py_DECREF(obj); return NULL; }

    /* Set all 7 attributes on the macmetalpy ndarray instance */
    int ok = 0;
    ok |= PyObject_SetAttr(obj, str_buffer,  Py_None);
    ok |= PyObject_SetAttr(obj, str_np_data, np_result);
    ok |= PyObject_SetAttr(obj, str_shape,   shape);
    ok |= PyObject_SetAttr(obj, str_dtype,   dtype);
    ok |= PyObject_SetAttr(obj, str_strides, strides);
    ok |= PyObject_SetAttr(obj, str_offset,  int_zero);
    ok |= PyObject_SetAttr(obj, str_base,    Py_None);

    Py_DECREF(strides);
    Py_DECREF(dtype);
    Py_DECREF(shape);

    if (ok != 0) {
        Py_DECREF(obj);
        return NULL;
    }
    return obj;
}

/* ------------------------------------------------------------------ */
/* Optimized wrap: tp_alloc + direct dict writes                      */
/* np_result MUST be a numpy ndarray                                  */
/* ------------------------------------------------------------------ */
static PyObject *
wrap_result_fast(PyObject *np_result, PyObject *dtype_override)
{
    PyArrayObject *arr = (PyArrayObject *)np_result;

    /* Allocate via tp_alloc — skips __new__ and __init__ overhead */
    PyObject *obj = cached_ndarray_type->tp_alloc(cached_ndarray_type, 0);
    if (!obj) return NULL;

    /* Build shape tuple from numpy array's C shape */
    PyObject *shape = shape_tuple_from_ndarray(arr);
    if (!shape) { Py_DECREF(obj); return NULL; }

    /* Determine dtype */
    PyObject *dtype;
    if (dtype_override) {
        dtype = dtype_override;
        Py_INCREF(dtype);
    } else {
        PyArray_Descr *descr = PyArray_DESCR(arr);
        dtype = (PyObject *)descr;
        Py_INCREF(dtype);
    }

    /* Compute element strides from shape */
    PyObject *strides = compute_c_strides_from_ndarray(arr);
    if (!strides) { Py_DECREF(dtype); Py_DECREF(shape); Py_DECREF(obj); return NULL; }

    /* Set all 7 attributes via direct dict writes */
    int ok = 0;
    ok |= dict_write(obj, str_buffer,  Py_None);
    ok |= dict_write(obj, str_np_data, np_result);
    ok |= dict_write(obj, str_shape,   shape);
    ok |= dict_write(obj, str_dtype,   dtype);
    ok |= dict_write(obj, str_strides, strides);
    ok |= dict_write(obj, str_offset,  int_zero);
    ok |= dict_write(obj, str_base,    Py_None);

    Py_DECREF(strides);
    Py_DECREF(dtype);
    Py_DECREF(shape);

    if (ok != 0) {
        Py_DECREF(obj);
        return NULL;
    }
    return obj;
}

/* ------------------------------------------------------------------ */
/* Legacy: wrap_result(ndarray_type, np_array) -> ndarray             */
/* ------------------------------------------------------------------ */
static PyObject *
accel_wrap_result(PyObject *self, PyObject *args)
{
    PyObject *ndarray_type;
    PyObject *np_array;

    if (!PyArg_ParseTuple(args, "OO", &ndarray_type, &np_array))
        return NULL;

    if (!PyType_Check(ndarray_type)) {
        PyErr_SetString(PyExc_TypeError, "first argument must be a type");
        return NULL;
    }

    if (!PyArray_Check(np_array)) {
        PyErr_SetString(PyExc_TypeError, "second argument must be a numpy array");
        return NULL;
    }

    return wrap_result_impl((PyTypeObject *)ndarray_type, np_array);
}

/* ------------------------------------------------------------------ */
/* Legacy: binary_op_fast(a, b, np_func, threshold) -> ndarray | None */
/* ------------------------------------------------------------------ */
static PyObject *
accel_binary_op_fast(PyObject *self, PyObject *args)
{
    PyObject *a, *b, *np_func;
    Py_ssize_t threshold;

    if (!PyArg_ParseTuple(args, "OOOn", &a, &b, &np_func, &threshold))
        return NULL;

    /* Check type(a) == type(b) */
    if (Py_TYPE(a) != Py_TYPE(b))
        Py_RETURN_NONE;

    /* Check a._np_data is not None */
    PyObject *a_np = PyObject_GetAttr(a, str_np_data);
    if (!a_np) { PyErr_Clear(); Py_RETURN_NONE; }
    if (a_np == Py_None) { Py_DECREF(a_np); Py_RETURN_NONE; }

    /* Check b._np_data is not None */
    PyObject *b_np = PyObject_GetAttr(b, str_np_data);
    if (!b_np) { PyErr_Clear(); Py_DECREF(a_np); Py_RETURN_NONE; }
    if (b_np == Py_None) { Py_DECREF(b_np); Py_DECREF(a_np); Py_RETURN_NONE; }

    /* Verify they are numpy arrays */
    if (!PyArray_Check(a_np) || !PyArray_Check(b_np)) {
        Py_DECREF(b_np);
        Py_DECREF(a_np);
        Py_RETURN_NONE;
    }

    /* Check a._np_data.size < threshold */
    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_np);
    if (a_size >= threshold) {
        Py_DECREF(b_np);
        Py_DECREF(a_np);
        Py_RETURN_NONE;
    }

    /* Check a._dtype is b._dtype or a._dtype == b._dtype */
    PyObject *a_dtype = PyObject_GetAttr(a, str_dtype);
    PyObject *b_dtype = PyObject_GetAttr(b, str_dtype);
    if (!a_dtype || !b_dtype) {
        PyErr_Clear();
        Py_XDECREF(a_dtype);
        Py_XDECREF(b_dtype);
        Py_DECREF(b_np);
        Py_DECREF(a_np);
        Py_RETURN_NONE;
    }

    int same_dtype = (a_dtype == b_dtype);
    if (!same_dtype) {
        same_dtype = (PyObject_RichCompareBool(a_dtype, b_dtype, Py_EQ) == 1);
    }
    Py_DECREF(a_dtype);
    Py_DECREF(b_dtype);

    if (!same_dtype) {
        Py_DECREF(b_np);
        Py_DECREF(a_np);
        Py_RETURN_NONE;
    }

    /* All checks passed — call np_func(a._np_data, b._np_data) */
    PyObject *result = PyObject_CallFunctionObjArgs(np_func, a_np, b_np, NULL);
    Py_DECREF(b_np);
    Py_DECREF(a_np);

    if (!result) return NULL;

    /* Ensure result is a numpy array */
    if (!PyArray_Check(result)) {
        PyObject *arr = PyArray_FROM_O(result);
        Py_DECREF(result);
        if (!arr) return NULL;
        result = arr;
    }

    /* Wrap result into ndarray */
    PyObject *wrapped = wrap_result_impl(Py_TYPE(a), result);
    Py_DECREF(result);
    return wrapped;
}

/* ------------------------------------------------------------------ */
/* Legacy: unary_op_fast(a, np_func, threshold) -> ndarray | None     */
/* ------------------------------------------------------------------ */
static PyObject *
accel_unary_op_fast(PyObject *self, PyObject *args)
{
    PyObject *a, *np_func;
    Py_ssize_t threshold;

    if (!PyArg_ParseTuple(args, "OOn", &a, &np_func, &threshold))
        return NULL;

    /* Check a._np_data is not None */
    PyObject *a_np = PyObject_GetAttr(a, str_np_data);
    if (!a_np) { PyErr_Clear(); Py_RETURN_NONE; }
    if (a_np == Py_None) { Py_DECREF(a_np); Py_RETURN_NONE; }

    /* Verify it's a numpy array */
    if (!PyArray_Check(a_np)) {
        Py_DECREF(a_np);
        Py_RETURN_NONE;
    }

    /* Check a._np_data.size < threshold */
    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_np);
    if (a_size >= threshold) {
        Py_DECREF(a_np);
        Py_RETURN_NONE;
    }

    /* All checks passed — call np_func(a._np_data) */
    PyObject *result = PyObject_CallFunctionObjArgs(np_func, a_np, NULL);
    Py_DECREF(a_np);

    if (!result) return NULL;

    /* Ensure result is a numpy array */
    if (!PyArray_Check(result)) {
        PyObject *arr = PyArray_FROM_O(result);
        Py_DECREF(result);
        if (!arr) return NULL;
        result = arr;
    }

    /* Wrap result into ndarray */
    PyObject *wrapped = wrap_result_impl(Py_TYPE(a), result);
    Py_DECREF(result);
    return wrapped;
}

/* ------------------------------------------------------------------ */
/* Helper: populate one dispatch table from a Python list of tuples   */
/* Each tuple is (np_func, threshold, flags)                          */
/* ------------------------------------------------------------------ */
static int
populate_table(OpEntry *table, PyObject *list, const char *name)
{
    if (!PyList_Check(list)) {
        PyErr_Format(PyExc_TypeError, "%s must be a list", name);
        return -1;
    }

    Py_ssize_t n = PyList_GET_SIZE(list);
    if (n > MAX_OPS) {
        PyErr_Format(PyExc_ValueError, "%s has too many entries (%zd > %d)",
                     name, n, MAX_OPS);
        return -1;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);  /* borrowed */
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 3) {
            PyErr_Format(PyExc_TypeError,
                         "%s[%zd] must be a (np_func, threshold, flags) tuple",
                         name, i);
            return -1;
        }

        PyObject *func = PyTuple_GET_ITEM(item, 0);      /* borrowed */
        PyObject *thresh_obj = PyTuple_GET_ITEM(item, 1); /* borrowed */
        PyObject *flags_obj = PyTuple_GET_ITEM(item, 2);  /* borrowed */

        Py_ssize_t thresh = PyLong_AsSsize_t(thresh_obj);
        if (thresh == -1 && PyErr_Occurred()) return -1;

        int flags = (int)PyLong_AsLong(flags_obj);
        if (flags == -1 && PyErr_Occurred()) return -1;

        Py_XDECREF(table[i].np_func);
        Py_INCREF(func);
        table[i].np_func   = func;
        table[i].threshold = thresh;
        table[i].flags     = flags;
    }

    /* Clear remaining entries */
    for (Py_ssize_t i = n; i < MAX_OPS; i++) {
        Py_XDECREF(table[i].np_func);
        table[i].np_func   = NULL;
        table[i].threshold = 0;
        table[i].flags     = 0;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* init_dispatch(ndarray_type, bool_dtype, binary_list, unary_list,   */
/*               cmp_list)  — METH_FASTCALL                           */
/* One-time setup that populates the dispatch tables                  */
/* ------------------------------------------------------------------ */
static PyObject *
accel_init_dispatch(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        PyErr_Format(PyExc_TypeError,
                     "init_dispatch() takes 5 arguments (%zd given)", nargs);
        return NULL;
    }

    PyObject *ndarray_type = args[0];
    PyObject *bool_dtype   = args[1];
    PyObject *binary_list  = args[2];
    PyObject *unary_list   = args[3];
    PyObject *cmp_list     = args[4];

    if (!PyType_Check(ndarray_type)) {
        PyErr_SetString(PyExc_TypeError,
                        "first argument must be the ndarray type");
        return NULL;
    }

    /* Cache the ndarray type */
    Py_XDECREF((PyObject *)cached_ndarray_type);
    cached_ndarray_type = (PyTypeObject *)ndarray_type;
    Py_INCREF(ndarray_type);

    /* Cache the bool dtype */
    Py_XDECREF(cached_bool_dtype);
    cached_bool_dtype = bool_dtype;
    Py_INCREF(bool_dtype);

    /* Cache float64 dtype for fast comparison */
    Py_XDECREF(cached_float64_dtype);
    cached_float64_dtype = (PyObject *)PyArray_DescrFromType(NPY_FLOAT64);
    if (!cached_float64_dtype) return NULL;

    /* Populate all three tables */
    if (populate_table(binary_table, binary_list, "binary_list") < 0)
        return NULL;
    if (populate_table(unary_table, unary_list, "unary_list") < 0)
        return NULL;
    if (populate_table(cmp_table, cmp_list, "cmp_list") < 0)
        return NULL;

    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* fast_binary(a, b, op_id) -> ndarray | None  — METH_FASTCALL       */
/* ------------------------------------------------------------------ */
static PyObject *
accel_fast_binary(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError,
                     "fast_binary() takes 3 arguments (%zd given)", nargs);
        return NULL;
    }

    PyObject *a = args[0];
    PyObject *b = args[1];
    Py_ssize_t op_id = PyLong_AsSsize_t(args[2]);
    if (op_id == -1 && PyErr_Occurred()) return NULL;

    if (op_id < 0 || op_id >= MAX_OPS || !binary_table[op_id].np_func)
        Py_RETURN_NONE;

    OpEntry *entry = &binary_table[op_id];

    /* 1. Check type(a) == cached_ndarray_type and type(b) == cached_ndarray_type */
    if (Py_TYPE(a) != cached_ndarray_type || Py_TYPE(b) != cached_ndarray_type)
        Py_RETURN_NONE;

    /* 2. Get a._np_data via direct dict access */
    PyObject *a_np = dict_read(a, str_np_data);
    if (!a_np || a_np == Py_None)
        Py_RETURN_NONE;

    /* 3. Get b._np_data via direct dict access */
    PyObject *b_np = dict_read(b, str_np_data);
    if (!b_np || b_np == Py_None)
        Py_RETURN_NONE;

    /* 4. Verify both are numpy arrays */
    if (!PyArray_Check(a_np) || !PyArray_Check(b_np))
        Py_RETURN_NONE;

    /* 5. Check a._np_data.size < threshold */
    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_np);
    if (a_size >= entry->threshold)
        Py_RETURN_NONE;

    /* 6. Check a._dtype is b._dtype or a._dtype == b._dtype */
    PyObject *a_dtype = dict_read(a, str_dtype);
    PyObject *b_dtype = dict_read(b, str_dtype);
    if (!a_dtype || !b_dtype)
        Py_RETURN_NONE;

    /* Float64: fall through to Python CPU path */
    if (a_dtype == cached_float64_dtype || b_dtype == cached_float64_dtype)
        Py_RETURN_NONE;

    if (a_dtype != b_dtype) {
        int eq = PyObject_RichCompareBool(a_dtype, b_dtype, Py_EQ);
        if (eq != 1)
            Py_RETURN_NONE;
    }

    /* 7. Check FLAG_FLOAT_ONLY: if set, dtype.kind must be 'f' */
    if (entry->flags & FLAG_FLOAT_ONLY) {
        PyObject *kind = PyObject_GetAttr(a_dtype, str_kind);
        if (!kind) { PyErr_Clear(); Py_RETURN_NONE; }
        const char *kind_str = PyUnicode_AsUTF8(kind);
        int is_float = (kind_str && kind_str[0] == 'f');
        Py_DECREF(kind);
        if (!is_float)
            Py_RETURN_NONE;
    }

    /* 8. Call np_func(a_np, b_np) via vectorcall */
    PyObject *call_args[2] = { a_np, b_np };
    PyObject *result = PyObject_Vectorcall(
        entry->np_func, call_args, 2, NULL);
    if (!result) return NULL;

    /* Ensure result is a numpy array */
    if (!PyArray_Check(result)) {
        PyObject *arr = PyArray_FROM_O(result);
        Py_DECREF(result);
        if (!arr) return NULL;
        result = arr;
    }

    /* 9. Wrap result using tp_alloc + direct dict writes */
    PyObject *wrapped = wrap_result_fast(result, NULL);
    Py_DECREF(result);
    return wrapped;
}

/* ------------------------------------------------------------------ */
/* fast_unary(a, op_id) -> ndarray | None  — METH_FASTCALL           */
/* ------------------------------------------------------------------ */
static PyObject *
accel_fast_unary(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError,
                     "fast_unary() takes 2 arguments (%zd given)", nargs);
        return NULL;
    }

    PyObject *a = args[0];
    Py_ssize_t op_id = PyLong_AsSsize_t(args[1]);
    if (op_id == -1 && PyErr_Occurred()) return NULL;

    if (op_id < 0 || op_id >= MAX_OPS || !unary_table[op_id].np_func)
        Py_RETURN_NONE;

    OpEntry *entry = &unary_table[op_id];

    /* 1. Check type(a) == cached_ndarray_type */
    if (Py_TYPE(a) != cached_ndarray_type)
        Py_RETURN_NONE;

    /* 2. Get a._np_data via direct dict access */
    PyObject *a_np = dict_read(a, str_np_data);
    if (!a_np || a_np == Py_None)
        Py_RETURN_NONE;

    /* 3. Get a._dtype and check it's not complex64 (dtype.kind == 'c') */
    PyObject *a_dtype = dict_read(a, str_dtype);
    if (!a_dtype)
        Py_RETURN_NONE;

    /* Float64: fall through to Python CPU path */
    if (a_dtype == cached_float64_dtype)
        Py_RETURN_NONE;

    PyObject *kind = PyObject_GetAttr(a_dtype, str_kind);
    if (!kind) { PyErr_Clear(); Py_RETURN_NONE; }
    const char *kind_str = PyUnicode_AsUTF8(kind);
    int is_complex = (kind_str && kind_str[0] == 'c');
    Py_DECREF(kind);
    if (is_complex)
        Py_RETURN_NONE;

    /* 4. Verify _np_data is a numpy array */
    if (!PyArray_Check(a_np))
        Py_RETURN_NONE;

    /* 5. Check a._np_data.size < threshold */
    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_np);
    if (a_size >= entry->threshold)
        Py_RETURN_NONE;

    /* 6. Call np_func(a_np) via vectorcall */
    PyObject *call_args[1] = { a_np };
    PyObject *result = PyObject_Vectorcall(
        entry->np_func, call_args, 1, NULL);
    if (!result) return NULL;

    /* Ensure result is a numpy array */
    if (!PyArray_Check(result)) {
        PyObject *arr = PyArray_FROM_O(result);
        Py_DECREF(result);
        if (!arr) return NULL;
        result = arr;
    }

    /* 7. Wrap result */
    PyObject *wrapped = wrap_result_fast(result, NULL);
    Py_DECREF(result);
    return wrapped;
}

/* ------------------------------------------------------------------ */
/* fast_cmp(a, b, op_id) -> ndarray | None  — METH_FASTCALL          */
/* ------------------------------------------------------------------ */
static PyObject *
accel_fast_cmp(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        PyErr_Format(PyExc_TypeError,
                     "fast_cmp() takes 3 arguments (%zd given)", nargs);
        return NULL;
    }

    PyObject *a = args[0];
    PyObject *b = args[1];
    Py_ssize_t op_id = PyLong_AsSsize_t(args[2]);
    if (op_id == -1 && PyErr_Occurred()) return NULL;

    if (op_id < 0 || op_id >= MAX_OPS || !cmp_table[op_id].np_func)
        Py_RETURN_NONE;

    OpEntry *entry = &cmp_table[op_id];

    /* 1. Check type(a) == cached_ndarray_type and type(b) == cached_ndarray_type */
    if (Py_TYPE(a) != cached_ndarray_type || Py_TYPE(b) != cached_ndarray_type)
        Py_RETURN_NONE;

    /* 2. Get a._np_data via direct dict access */
    PyObject *a_np = dict_read(a, str_np_data);
    if (!a_np || a_np == Py_None)
        Py_RETURN_NONE;

    /* 3. Get b._np_data via direct dict access */
    PyObject *b_np = dict_read(b, str_np_data);
    if (!b_np || b_np == Py_None)
        Py_RETURN_NONE;

    /* 4. Verify both are numpy arrays */
    if (!PyArray_Check(a_np) || !PyArray_Check(b_np))
        Py_RETURN_NONE;

    /* 5. Check a._np_data.size < threshold */
    npy_intp a_size = PyArray_SIZE((PyArrayObject *)a_np);
    if (a_size >= entry->threshold)
        Py_RETURN_NONE;

    /* 6. Check a._dtype is b._dtype or a._dtype == b._dtype */
    PyObject *a_dtype = dict_read(a, str_dtype);
    PyObject *b_dtype = dict_read(b, str_dtype);
    if (!a_dtype || !b_dtype)
        Py_RETURN_NONE;

    /* Float64: fall through to Python CPU path */
    if (a_dtype == cached_float64_dtype || b_dtype == cached_float64_dtype)
        Py_RETURN_NONE;

    if (a_dtype != b_dtype) {
        int eq = PyObject_RichCompareBool(a_dtype, b_dtype, Py_EQ);
        if (eq != 1)
            Py_RETURN_NONE;
    }

    /* 7. Call np_func(a_np, b_np) via vectorcall */
    PyObject *call_args[2] = { a_np, b_np };
    PyObject *result = PyObject_Vectorcall(
        entry->np_func, call_args, 2, NULL);
    if (!result) return NULL;

    /* Ensure result is a numpy array */
    if (!PyArray_Check(result)) {
        PyObject *arr = PyArray_FROM_O(result);
        Py_DECREF(result);
        if (!arr) return NULL;
        result = arr;
    }

    /* 8. Wrap result with cached_bool_dtype override */
    PyObject *wrapped = wrap_result_fast(result, cached_bool_dtype);
    Py_DECREF(result);
    return wrapped;
}

/* ------------------------------------------------------------------ */
/* Module method table                                                */
/* ------------------------------------------------------------------ */
static PyMethodDef accel_methods[] = {
    /* Legacy functions (backward compatible) */
    {"wrap_result",    accel_wrap_result,    METH_VARARGS,
     "wrap_result(ndarray_type, np_array) -> ndarray\n"
     "Fast construction of an ndarray from a numpy result."},
    {"binary_op_fast", accel_binary_op_fast, METH_VARARGS,
     "binary_op_fast(a, b, np_func, threshold) -> ndarray | None\n"
     "Fused binary-op check + compute + wrap. Returns None on fast-path miss."},
    {"unary_op_fast",  accel_unary_op_fast,  METH_VARARGS,
     "unary_op_fast(a, np_func, threshold) -> ndarray | None\n"
     "Fused unary-op check + compute + wrap. Returns None on fast-path miss."},

    /* New dispatch-table functions (METH_FASTCALL) */
    {"init_dispatch",  (PyCFunction)accel_init_dispatch, METH_FASTCALL,
     "init_dispatch(ndarray_type, bool_dtype, binary_list, unary_list, cmp_list)\n"
     "One-time setup that populates the dispatch tables."},
    {"fast_binary",    (PyCFunction)accel_fast_binary,   METH_FASTCALL,
     "fast_binary(a, b, op_id) -> ndarray | None\n"
     "Dispatch-table binary op. Returns None on fast-path miss."},
    {"fast_unary",     (PyCFunction)accel_fast_unary,    METH_FASTCALL,
     "fast_unary(a, op_id) -> ndarray | None\n"
     "Dispatch-table unary op. Returns None on fast-path miss."},
    {"fast_cmp",       (PyCFunction)accel_fast_cmp,      METH_FASTCALL,
     "fast_cmp(a, b, op_id) -> ndarray | None\n"
     "Dispatch-table comparison op. Returns None on fast-path miss."},

    {NULL, NULL, 0, NULL}
};

/* ------------------------------------------------------------------ */
/* Module definition                                                  */
/* ------------------------------------------------------------------ */
static struct PyModuleDef accelerator_module = {
    PyModuleDef_HEAD_INIT,
    "_accelerator",
    "C-accelerated hot paths for macmetalpy ndarray operations.",
    -1,
    accel_methods
};

/* ------------------------------------------------------------------ */
/* Module init — intern strings, import numpy                         */
/* ------------------------------------------------------------------ */
PyMODINIT_FUNC
PyInit__accelerator(void)
{
    /* Import numpy C API */
    import_array();

    /* Intern attribute name strings (private names on macmetalpy.ndarray) */
    str_buffer  = PyUnicode_InternFromString("_buffer");
    str_np_data = PyUnicode_InternFromString("_np_data");
    str_shape   = PyUnicode_InternFromString("_shape");
    str_dtype   = PyUnicode_InternFromString("_dtype");
    str_strides = PyUnicode_InternFromString("_strides");
    str_offset  = PyUnicode_InternFromString("_offset");
    str_base    = PyUnicode_InternFromString("_base");
    str_kind    = PyUnicode_InternFromString("kind");

    if (!str_buffer || !str_np_data || !str_shape || !str_dtype ||
        !str_strides || !str_offset || !str_base || !str_kind) {
        return NULL;
    }

    /* Cache empty tuple and int(0) */
    empty_tuple = PyTuple_New(0);
    if (!empty_tuple) return NULL;

    int_zero = PyLong_FromLong(0);
    if (!int_zero) return NULL;

    /* Initialize dispatch tables to zero */
    memset(binary_table, 0, sizeof(binary_table));
    memset(unary_table,  0, sizeof(unary_table));
    memset(cmp_table,    0, sizeof(cmp_table));

    PyObject *m = PyModule_Create(&accelerator_module);
    return m;
}
