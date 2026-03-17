from __future__ import annotations

from typing import Any, Iterable, Mapping


class ExamplesTable:
    """Duck-typed wrapper around tabular training examples.

    The goal is not to define a strict dataframe protocol, but to accept the
    common shapes users already have:

    - pandas DataFrame
    - polars DataFrame
    - duckdb relation
    - pyarrow Table / RecordBatch
    - pyspark DataFrame
    - numpy structured array / recarray
    - list/iterable of dict records

    Internally this wrapper only needs four things:

    - column names
    - row count
    - record serialization
    - coarse type inference per column

    Examples
    --------
    Wrap a simple list of records:

    >>> table = ExamplesTable([{"question": "Q", "score": 1.0}])
    >>> table.columns
    ['question', 'score']
    >>> len(table)
    1

    Convert the table back into normalized records:

    >>> table.to_records()
    [{'question': 'Q', 'score': 1.0}]
    """

    __slots__ = ("raw",)

    def __init__(self, raw: Any):
        """Store a tabular object or iterable of record mappings.

        Parameters
        ----------
        raw : Any
            Dataframe-like object or iterable of records.

        Examples
        --------
        >>> table = ExamplesTable([{"a": 1}, {"a": 2}])
        >>> table.raw
        [{'a': 1}, {'a': 2}]
        """
        self.raw = raw

    @property
    def columns(self) -> list[str]:
        """Return the detected column names.

        Returns
        -------
        list[str]
            Column names inferred from schema information or record keys.

        Examples
        --------
        >>> ExamplesTable([{"question": "Q", "answer": "A"}]).columns
        ['question', 'answer']
        """
        raw = self.raw
        if hasattr(raw, "columns"):
            return [str(column) for column in raw.columns]
        if hasattr(raw, "column_names"):
            return [str(column) for column in raw.column_names]
        if hasattr(raw, "dtype") and getattr(raw.dtype, "names", None):
            return [str(column) for column in raw.dtype.names]
        if hasattr(raw, "schema"):
            schema = raw.schema
            if isinstance(schema, Mapping):
                return [str(column) for column in schema.keys()]
            if hasattr(schema, "names"):
                return [str(column) for column in schema.names]
            if hasattr(schema, "fieldNames"):
                return [str(column) for column in schema.fieldNames()]
        if hasattr(raw, "dtypes"):
            try:
                return [str(column) for column, _ in raw.dtypes]
            except Exception:  # noqa: BLE001
                pass
        records = self.to_records()
        return list(records[0].keys()) if records else []

    def __len__(self) -> int:
        """Return the number of rows.

        Returns
        -------
        int
            Number of example rows.

        Examples
        --------
        >>> len(ExamplesTable([{"a": 1}, {"a": 2}, {"a": 3}]))
        3
        """
        raw = self.raw
        if hasattr(raw, "height"):
            return int(raw.height)
        if hasattr(raw, "num_rows") and raw.num_rows is not None:
            return int(raw.num_rows)
        if hasattr(raw, "count"):
            try:
                count = raw.count()
                if isinstance(count, int):
                    return count
            except Exception:  # noqa: BLE001
                pass
        try:
            return len(raw)
        except TypeError:
            return len(self.to_records())

    def infer_type(self, name: str) -> type:
        """Infer a coarse Python type for one column.

        Parameters
        ----------
        name : str
            Column name to inspect.

        Returns
        -------
        type
            Inferred Python type, defaulting to ``str`` when no better guess is
            available.

        Examples
        --------
        >>> table = ExamplesTable([{"score": 1.0, "ok": True, "meta": {"x": 1}}])
        >>> table.infer_type("score") is float
        True
        >>> table.infer_type("ok") is bool
        True
        >>> table.infer_type("meta") is dict
        True
        >>> table.infer_type("missing") is str
        True
        """
        schema_type = _infer_type_from_schema(self.raw, name)
        if schema_type is not None:
            return schema_type

        for record in self.to_records():
            if name not in record:
                continue
            value = record[name]
            if value is None:
                continue
            return _infer_python_value_type(value)
        return str

    def to_records(self) -> list[dict[str, Any]]:
        """Normalize the underlying data into record dictionaries.

        Returns
        -------
        list[dict[str, Any]]
            Example rows represented as plain Python dictionaries.

        Examples
        --------
        >>> ExamplesTable([{"question": "Q1"}, {"question": "Q2"}]).to_records()
        [{'question': 'Q1'}, {'question': 'Q2'}]
        """
        raw = self.raw

        if hasattr(raw, "to_dicts"):
            records = raw.to_dicts()
        elif hasattr(raw, "to_pylist"):
            records = raw.to_pylist()
        elif hasattr(raw, "to_dict"):
            try:
                records = raw.to_dict("records")
            except TypeError:
                records = None
            if records is None:
                raise TypeError("Unsupported tabular object: .to_dict() exists but does not support record output.")
        elif hasattr(raw, "arrow"):
            table = raw.arrow()
            if hasattr(table, "to_pylist"):
                records = table.to_pylist()
            else:
                raise TypeError("Unsupported tabular object: .arrow() result cannot be converted to records.")
        elif hasattr(raw, "to_arrow"):
            table = raw.to_arrow()
            if hasattr(table, "to_pylist"):
                records = table.to_pylist()
            else:
                raise TypeError("Unsupported tabular object: .to_arrow() result cannot be converted to records.")
        elif hasattr(raw, "toLocalIterator"):
            records = list(raw.toLocalIterator())
        elif hasattr(raw, "collect") and hasattr(raw, "dtypes"):
            records = raw.collect()
        elif hasattr(raw, "df"):
            df = raw.df()
            if hasattr(df, "to_dict"):
                records = df.to_dict("records")
            else:
                raise TypeError("Unsupported tabular object: .df() result cannot be converted to records.")
        elif hasattr(raw, "dtype") and getattr(raw.dtype, "names", None):
            records = []
            names = list(raw.dtype.names)
            for row in raw:
                records.append({name: row[name].item() if hasattr(row[name], "item") else row[name] for name in names})
        elif isinstance(raw, Iterable):
            records = list(raw)
        else:
            raise TypeError(
                "Examples must be dataframe-like (pandas/polars/duckdb/arrow/pyspark/numpy-structured) or an iterable of dict records."
            )

        normalized: list[dict[str, Any]] = []
        for record in records:
            if isinstance(record, Mapping):
                normalized.append(dict(record))
                continue
            if hasattr(record, "asDict"):
                normalized.append(dict(record.asDict(recursive=True)))
                continue
            raise TypeError("Example rows must be mapping-like records.")
        return normalized


def normalize_examples(data: Any | None) -> ExamplesTable | None:
    """Normalize training examples into an :class:`ExamplesTable`.

    Parameters
    ----------
    data : Any | None
        Input examples, an existing :class:`ExamplesTable`, or ``None``.

    Returns
    -------
    ExamplesTable | None
        Wrapped examples, or ``None`` when no examples were provided.

    Examples
    --------
    >>> normalize_examples(None) is None
    True
    >>> isinstance(normalize_examples([{"a": 1}]), ExamplesTable)
    True
    >>> table = ExamplesTable([{"a": 1}])
    >>> normalize_examples(table) is table
    True
    """
    if data is None:
        return None
    if isinstance(data, ExamplesTable):
        return data
    return ExamplesTable(data)



def infer_type(data: ExamplesTable | None, name: str) -> type:
    """Infer a column type from normalized examples.

    Parameters
    ----------
    data : ExamplesTable | None
        Normalized example table.
    name : str
        Column name to inspect.

    Returns
    -------
    type
        Inferred Python type, or ``str`` when the column is unavailable.

    Examples
    --------
    >>> data = normalize_examples([{"score": 1.0, "label": "good"}])
    >>> infer_type(data, "score") is float
    True
    >>> infer_type(data, "label") is str
    True
    >>> infer_type(data, "missing") is str
    True
    """
    if data is None or name not in data.columns:
        return str
    return data.infer_type(name)



def _infer_type_from_schema(raw: Any, name: str) -> type | None:
    # pandas-like: df[col].dtype.kind
    if hasattr(raw, "__getitem__") and hasattr(raw, "columns") and name in getattr(raw, "columns"):
        try:
            dtype = raw[name].dtype
            kind = getattr(dtype, "kind", None)
            if kind is not None:
                return {"f": float, "i": int, "u": int, "b": bool}.get(kind, str)
        except Exception:  # noqa: BLE001
            pass

    # numpy structured arrays / recarrays
    if hasattr(raw, "dtype") and getattr(raw.dtype, "names", None) and name in raw.dtype.names:
        try:
            inferred = _infer_type_from_dtype_name(str(raw.dtype[name]))
            if inferred is not None:
                return inferred
        except Exception:  # noqa: BLE001
            pass

    # polars-like / arrow-like / spark-like schema objects
    if hasattr(raw, "schema"):
        schema = raw.schema
        if isinstance(schema, Mapping) and name in schema:
            inferred = _infer_type_from_dtype_name(str(schema[name]))
            if inferred is not None:
                return inferred
        if hasattr(schema, "field"):
            try:
                inferred = _infer_type_from_dtype_name(str(schema.field(name).type))
                if inferred is not None:
                    return inferred
            except Exception:  # noqa: BLE001
                pass
        if hasattr(schema, "__iter__"):
            try:
                for field in schema:
                    field_name = getattr(field, "name", None)
                    if field_name == name:
                        dtype = getattr(field, "dataType", None) or getattr(field, "type", None)
                        inferred = _infer_type_from_dtype_name(str(dtype))
                        if inferred is not None:
                            return inferred
            except Exception:  # noqa: BLE001
                pass

    # pyspark-like: dtypes is a list of (name, type)
    if hasattr(raw, "dtypes"):
        try:
            for column, dtype in raw.dtypes:
                if column == name:
                    inferred = _infer_type_from_dtype_name(str(dtype))
                    if inferred is not None:
                        return inferred
        except Exception:  # noqa: BLE001
            pass

    # duckdb-like: relation.columns + relation.types
    if hasattr(raw, "columns") and hasattr(raw, "types"):
        try:
            columns = list(raw.columns)
            index = columns.index(name)
            inferred = _infer_type_from_dtype_name(str(raw.types[index]))
            if inferred is not None:
                return inferred
        except Exception:  # noqa: BLE001
            pass

    return None



def _infer_type_from_dtype_name(dtype_name: str) -> type | None:
    text = dtype_name.upper()
    if any(token in text for token in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "FLOAT16", "FLOAT32", "FLOAT64")):
        return float
    if any(
        token in text
        for token in (
            "INT",
            "INTEGER",
            "BIGINT",
            "SMALLINT",
            "TINYINT",
            "UBIGINT",
            "UINTEGER",
            "INT8",
            "INT16",
            "INT32",
            "INT64",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "LONG",
            "SHORT",
            "BYTE",
        )
    ):
        return int
    if any(token in text for token in ("BOOL", "BOOLEAN")):
        return bool
    if any(token in text for token in ("LIST", "ARRAY")):
        return list
    if any(token in text for token in ("STRUCT", "MAP", "DICT", "OBJECT")):
        return dict
    if any(token in text for token in ("STR", "TEXT", "VARCHAR", "UTF8", "STRING", "UNICODE")):
        return str
    return None



def _infer_python_value_type(value: Any) -> type:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, str):
        return str
    if isinstance(value, (list, tuple, set)):
        return list
    if isinstance(value, Mapping):
        return dict
    return type(value)
