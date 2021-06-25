# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:32:07 2021

@author: kpapke
"""
import os
import tables
import numpy

from ..log import logmanager

logger = logmanager.getLogger(__name__)

__all__ = ['HDFStore']


class HDFStore:
    """
    Dict-like IO interface for storing datasets in HDF5 files.

    Parameters
    ----------
    path : str
        File path to HDF5 file.
    mode : {'a', 'w', 'r', 'r+'}, default 'a'
        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        As of v0.20.2 these additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
         'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.
    fletcher32 : bool, default False
        If applying compression use the fletcher32 checksum.
    **kwargs
        These parameters will be passed to the PyTables open_file method.

    Examples
    --------
    Writing Dataframe to hdf5 file

    >>> bar = pd.DataFrame(np.random.randn(3), columns=['A', 'B', 'C'])
    >>> store = hsi.HDFStore('test.h5')
    >>> store['foo'] = bar   # write to HDF5
    >>> bar = store['foo']   # retrieve
    >>> store.close()


    """

    _handle: Optional["File"]
    _mode: str
    _complevel: int
    _fletcher32: bool

    def __init__(
        self,
        path,
        mode: str = "a",
        complevel: Optional[int] = None,
        complib=None,
        fletcher32: bool = False,
        **kwargs,
    ):

        if "hsformat" in kwargs:
            raise ValueError("hsformat is not a defined argument for HDFStore")

        tables = import_optional_dependency("tables")

        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(
                f"complib only supports {tables.filters.all_complibs} compression."
            )

        if complib is None and complevel is not None:
            complib = tables.filters.default_complib

        self._path = stringify_path(path)
        if mode is None:
            mode = "a"
        self._mode = mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self):
        return self._path

    @property
    def root(self):
        """ return the root node """
        self._check_if_open()
        assert self._handle is not None  # for mypy
        return self._handle.root

    @property
    def filename(self):
        return self._path

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value):
        self.put(key, value)

    def __delitem__(self, key: str):
        return self.remove(key)

    def __getattr__(self, name: str):
        """ allow attribute access to get stores """
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __contains__(self, key: str) -> bool:
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        node = self.get_node(key)
        if node is not None:
            name = node._v_pathname
            if name == key or name[1:] == key:
                return True
        return False

    def __len__(self) -> int:
        return len(self.groups())

    def __repr__(self) -> str:
        pstr = pprint_thing(self._path)
        return f"{type(self)}\nFile path: {pstr}\n"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def keys(self, include: str = "pandas") -> List[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.
        Parameters
        ----------
        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.
                .. versionadded:: 1.1.0
        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').
        Raises
        ------
        raises ValueError if kind has an illegal value
        """
        if include == "pandas":
            return [n._v_pathname for n in self.groups()]

        elif include == "native":
            assert self._handle is not None  # mypy
            return [
                n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")
            ]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        """
        iterate on key->group
        """
        for g in self.groups():
            yield g._v_pathname, g

    iteritems = items

    def open(self, mode: str = "a", **kwargs):
        """
        Open the file in the specified mode
        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        if self._mode != mode:
            # if we are changing a write mode to read, ok
            if self._mode in ["a", "w"] and mode in ["r", "r+"]:
                pass
            elif mode in ["w"]:
                # this would truncate, raise here
                if self.is_open:
                    raise Exception(
                        f"Re-opening the file [{self._path}] with mode [{self._mode}] "
                        "will delete the current file!"
                    )

            self._mode = mode

        # close and reopen the handle
        if self.is_open:
            self.close()

        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self):
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync: bool = False):
        """
        Force all buffered modifications to be written to disk.
        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.
        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key: str):
        """
        Retrieve pandas object stored in file.
        Parameters
        ----------
        key : str
        Returns
        -------
        object
            Same type as object stored in file.
        """
        with patch_pickle():
            # GH#31167 Without this patch, pickle doesn't know how to unpickle
            #  old DateOffset objects now that they are cdef classes.
            group = self.get_node(key)
            if group is None:
                raise KeyError(f"No object named {key} in the file")
            return self._read_group(group)

    def select(
        self,
        key: str,
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator=False,
        chunksize=None,
        auto_close: bool = False,
    ):
        """
        Retrieve pandas object stored in file, optionally based on where criteria.
        .. warning::
           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" hsformat.
           Loading pickled data received from untrusted sources can be unsafe.
           See: https://docs.python.org/3/library/pickle.html for more.
        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.
        Returns
        -------
        object
            Retrieved object from file.
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        # create the storer and axes
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        # function to call on iteration
        def func(_start, _stop, _where):
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result()

    def select_as_coordinates(
        self,
        key: str,
        where=None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        """
        return the selection as an Index
        .. warning::
           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" hsformat.
           Loading pickled data received from untrusted sources can be unsafe.
           See: https://docs.python.org/3/library/pickle.html for more.
        Parameters
        ----------
        key : str
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        """
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_coordinates with a table")
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(
        self,
        key: str,
        column: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        """
        return a single column from the table. This is generally only useful to
        select an indexable
        .. warning::
           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" hsformat.
           Loading pickled data received from untrusted sources can be unsafe.
           See: https://docs.python.org/3/library/pickle.html for more.
        Parameters
        ----------
        key : str
        column : str
            The column of interest.
        start : int or None, default None
        stop : int or None, default None
        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)
        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_column with a table")
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(
        self,
        keys,
        where=None,
        selector=None,
        columns=None,
        start=None,
        stop=None,
        iterator=False,
        chunksize=None,
        auto_close: bool = False,
    ):
        """
        Retrieve pandas objects from multiple tables.
        .. warning::
           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" hsformat.
           Loading pickled data received from untrusted sources can be unsafe.
           See: https://docs.python.org/3/library/pickle.html for more.
        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : boolean, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.
        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        """
        # default to single select
        where = _ensure_term(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(
                key=keys,
                where=where,
                columns=columns,
                start=start,
                stop=stop,
                iterator=iterator,
                chunksize=chunksize,
                auto_close=auto_close,
            )

        if not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a list/tuple")

        if not len(keys):
            raise ValueError("keys must have a non-zero length")

        if selector is None:
            selector = keys[0]

        # collect the tables
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)

        # validate rows
        nrows = None
        for t, k in itertools.chain([(s, selector)], zip(tbls, keys)):
            if t is None:
                raise KeyError(f"Invalid table [{k}]")
            if not t.is_table:
                raise TypeError(
                    f"object [{t.pathname}] is not a table, and cannot be used in all "
                    "select as multiple"
                )

            if nrows is None:
                nrows = t.nrows
            elif t.nrows != nrows:
                raise ValueError("all tables must have exactly the same nrows!")

        # The isinstance checks here are redundant with the check above,
        #  but necessary for mypy; see GH#29757
        _tbls = [x for x in tbls if isinstance(x, Table)]

        # axis is the concentration axes
        axis = list({t.non_index_axes[0][0] for t in _tbls})[0]

        def func(_start, _stop, _where):

            # retrieve the objs, _where is always passed as a set of
            # coordinates here
            objs = [
                t.read(where=_where, columns=columns, start=_start, stop=_stop)
                for t in tbls
            ]

            # concat and return
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result(coordinates=True)

    def put(
        self,
        key: str,
        value: FrameOrSeries,
        format=None,
        index=True,
        append=False,
        complib=None,
        complevel: Optional[int] = None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep=None,
        data_columns: Optional[List[str]] = None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
        dropna: bool = False,
    ):
        """
        Store object in HDFStore.
        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : 'fixed(f)|table(t)', default is 'fixed'
            Format to use when storing object in HDFStore. Value can be one of:
            ``'fixed'``
                Fixed hsformat.  Fast writing/reading. Not-appendable, nor searchable.
            ``'table'``
                Table hsformat.  Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        append : bool, default False
            This will force Table hsformat, append the input data to the existing.
        data_columns : list, default None
            List of columns to create as data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str, default None
            Provide an encoding for strings.
        track_times : bool, default True
            Parameter is propagated to 'create_table' method of 'PyTables'.
            If set to False it enables to have the same h5 files (same hashes)
            independent on creation time.
            .. versionadded:: 1.1.0
        """
        if format is None:
            format = get_option("io.hdf.default_format") or "fixed"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
            dropna=dropna,
        )

    def remove(self, key: str, where=None, start=None, stop=None):
        """
        Remove pandas object partially by specifying the where condition
        Parameters
        ----------
        key : string
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        Returns
        -------
        number of rows removed (or None if not a Table)
        Raises
        ------
        raises KeyError if key is not a valid store
        """
        where = _ensure_term(where, scope_level=1)
        try:
            s = self.get_storer(key)
        except KeyError:
            # the key is not a valid store, re-raising KeyError
            raise
        except AssertionError:
            # surface any assertion errors for e.g. debugging
            raise
        except Exception as err:
            # In tests we get here with ClosedFileError, TypeError, and
            #  _table_mod.NoSuchNodeError.  TODO: Catch only these?

            if where is not None:
                raise ValueError(
                    "trying to remove a node with a non-None where clause!"
                ) from err

            # we are actually trying to remove a node (with children)
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None

        # remove the node
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)

        # delete from the table
        else:
            if not s.is_table:
                raise ValueError(
                    "can only remove with where on objects written as tables"
                )
            return s.delete(where=where, start=start, stop=stop)

    def append(
        self,
        key: str,
        value: FrameOrSeries,
        format=None,
        axes=None,
        index=True,
        append=True,
        complib=None,
        complevel: Optional[int] = None,
        columns=None,
        min_itemsize: Optional[Union[int, Dict[str, int]]] = None,
        nan_rep=None,
        chunksize=None,
        expectedrows=None,
        dropna: Optional[bool] = None,
        data_columns: Optional[List[str]] = None,
        encoding=None,
        errors: str = "strict",
    ):
        """
        Append to Table in file. Node must already exist and be Table
        hsformat.
        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : 'table' is the default
            Format to use when storing object in HDFStore.  Value can be one of:
            ``'table'``
                Table hsformat. Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        append       : bool, default True
            Append the input data to the existing.
        data_columns : list of columns, or True, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        min_itemsize : dict of columns that specify minimum str sizes
        nan_rep      : str to use as str nan representation
        chunksize    : size to chunk the writing
        expectedrows : expected TOTAL row size of this table
        encoding     : default None, provide an encoding for str
        dropna : bool, default False
            Do not write an ALL nan row to the store settable
            by the option 'io.hdf.dropna_table'.
        Notes
        -----
        Does *not* check if data being appended overlaps with existing
        data in the table, so be careful
        """
        if columns is not None:
            raise TypeError(
                "columns is not a supported keyword in append, try data_columns"
            )

        if dropna is None:
            dropna = get_option("io.hdf.dropna_table")
        if format is None:
            format = get_option("io.hdf.default_format") or "table"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            axes=axes,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
        )

    def append_to_multiple(
        self,
        d: Dict,
        value,
        selector,
        data_columns=None,
        axes=None,
        dropna=False,
        **kwargs,
    ):
        """
        Append to multiple tables
        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : list of columns to create as data columns, or True to
            use all columns
        dropna : if evaluates to True, drop rows from all tables if any single
                 row in each table has all NaN. Default False.
        Notes
        -----
        axes parameter is currently not accepted
        """
        if axes is not None:
            raise TypeError(
                "axes is currently not accepted as a parameter to append_to_multiple; "
                "you can create the tables independently instead"
            )

        if not isinstance(d, dict):
            raise ValueError(
                "append_to_multiple must have a dictionary specified as the "
                "way to split the value"
            )

        if selector not in d:
            raise ValueError(
                "append_to_multiple requires a selector that is in passed dict"
            )

        # figure out the splitting axis (the non_index_axis)
        axis = list(set(range(value.ndim)) - set(_AXES_MAP[type(value)]))[0]

        # figure out how to split the value
        remain_key = None
        remain_values: List = []
        for k, v in d.items():
            if v is None:
                if remain_key is not None:
                    raise ValueError(
                        "append_to_multiple can only have one value in d that is None"
                    )
                remain_key = k
            else:
                remain_values.extend(v)
        if remain_key is not None:
            ordered = value.axes[axis]
            ordd = ordered.difference(Index(remain_values))
            ordd = sorted(ordered.get_indexer(ordd))
            d[remain_key] = ordered.take(ordd)

        # data_columns
        if data_columns is None:
            data_columns = d[selector]

        # ensure rows are synchronized across the tables
        if dropna:
            idxs = (value[cols].dropna(how="all").index for cols in d.values())
            valid_index = next(idxs)
            for index in idxs:
                valid_index = valid_index.intersection(index)
            value = value.loc[valid_index]

        min_itemsize = kwargs.pop("min_itemsize", None)

        # append
        for k, v in d.items():
            dc = data_columns if k == selector else None

            # compute the val
            val = value.reindex(v, axis=axis)

            filtered = (
                {key: value for (key, value) in min_itemsize.items() if key in v}
                if min_itemsize is not None
                else None
            )
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def create_table_index(
        self,
        key: str,
        columns=None,
        optlevel: Optional[int] = None,
        kind: Optional[str] = None,
    ):
        """
        Create a pytables index on the table.
        Parameters
        ----------
        key : str
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.
            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.
        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".
        Raises
        ------
        TypeError: raises if the node is not a table
        """
        # version requirements
        _tables()
        s = self.get_storer(key)
        if s is None:
            return

        if not isinstance(s, Table):
            raise TypeError("cannot create table index on a Fixed hsformat store")
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self):
        """
        Return a list of all the top-level nodes.
        Each node returned is not a pandas storage object.
        Returns
        -------
        list
            List of objects.
        """
        # _tables()
        # self._check_if_open()
        # assert self._handle is not None  # for mypy
        # assert _table_mod is not None  # for mypy
        # return [
        #     g
        #     for g in self._handle.walk_groups()
        #     if (
        #         not isinstance(g, _table_mod.link.Link)
        #         and (
        #             getattr(g._v_attrs, "pandas_type", None)
        #             or getattr(g, "table", None)
        #             or (isinstance(g, _table_mod.table.Table) and g._v_name != "table")
        #         )
        #     )
        # ]

    # def walk(self, where="/"):
    #     """
    #     Walk the pytables group hierarchy for pandas objects.
    #     This generator will yield the group path, subgroups and pandas object
    #     names for each group.
    #     Any non-pandas PyTables objects that are not a group will be ignored.
    #     The `where` group itself is listed first (preorder), then each of its
    #     child groups (following an alphanumerical order) is also traversed,
    #     following the same procedure.
    #     .. versionadded:: 0.24.0
    #     Parameters
    #     ----------
    #     where : str, default "/"
    #         Group where to start walking.
    #     Yields
    #     ------
    #     path : str
    #         Full path to a group (without trailing '/').
    #     groups : list
    #         Names (strings) of the groups contained in `path`.
    #     leaves : list
    #         Names (strings) of the pandas objects contained in `path`.
    #     """
    #     _tables()
    #     self._check_if_open()
    #     assert self._handle is not None  # for mypy
    #     assert _table_mod is not None  # for mypy
    #
    #     for g in self._handle.walk_groups(where):
    #         if getattr(g._v_attrs, "pandas_type", None) is not None:
    #             continue
    #
    #         groups = []
    #         leaves = []
    #         for child in g._v_children.values():
    #             pandas_type = getattr(child._v_attrs, "pandas_type", None)
    #             if pandas_type is None:
    #                 if isinstance(child, _table_mod.group.Group):
    #                     groups.append(child._v_name)
    #             else:
    #                 leaves.append(child._v_name)
    #
    #         yield (g._v_pathname.rstrip("/"), groups, leaves)
    #
    # def get_node(self, key: str) -> Optional["Node"]:
    #     """ return the node with the key or None if it does not exist """
    #     self._check_if_open()
    #     if not key.startswith("/"):
    #         key = "/" + key
    #
    #     assert self._handle is not None
    #     assert _table_mod is not None  # for mypy
    #     try:
    #         node = self._handle.get_node(self.root, key)
    #     except _table_mod.exceptions.NoSuchNodeError:
    #         return None
    #
    #     assert isinstance(node, _table_mod.Node), type(node)
    #     return node
    #
    # def get_storer(self, key: str) -> Union["GenericFixed", "Table"]:
    #     """ return the storer object for a key, raise if not in the file """
    #     group = self.get_node(key)
    #     if group is None:
    #         raise KeyError(f"No object named {key} in the file")
    #
    #     s = self._create_storer(group)
    #     s.infer_axes()
    #     return s


    def info(self):
        """ Print detailed information on the store.

        Returns
        -------
        str
        """
        path = self._path
        output = f"{type(self)}\nFile path: {path}\n"

        if self.is_open:
            lkeys = sorted(self.keys())
            if len(lkeys):
                keys = []
                values = []

                # for k in lkeys:
                #     try:
                #         s = self.get_storer(k)
                #         if s is not None:
                #             keys.append(pprint_thing(s.pathname or k))
                #             values.append(pprint_thing(s or "invalid_HDFStore node"))
                #     except AssertionError:
                #         # surface any assertion errors for e.g. debugging
                #         raise
                #     except Exception as detail:
                #         keys.append(k)
                #         dstr = pprint_thing(detail)
                #         values.append(f"[invalid_HDFStore node: {dstr}]")
                #
                # output += adjoin(12, keys, values)
            else:
                output += "Empty"
        else:
            output += "File is CLOSED"

        return output

    # ------------------------------------------------------------------------
    # private methods

    # def _check_if_open(self):
    #     if not self.is_open:
    #         raise ClosedFileError(f"{self._path} file is not open!")
    #
    # def _validate_format(self, hsformat: str) -> str:
    #     """ validate / deprecate formats """
    #     # validate
    #     try:
    #         hsformat = _FORMAT_MAP[hsformat.lower()]
    #     except KeyError as err:
    #         raise TypeError(f"invalid HDFStore hsformat specified [{hsformat}]") from err
    #
    #     return hsformat





