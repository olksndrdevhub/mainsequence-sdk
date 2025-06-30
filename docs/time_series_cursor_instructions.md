# Instructions to create TimeSeries

This instructions specifies the basic configurations and best practices for extendint the `TimeSerie` class
in the main sequence platform

* When possible always introspect the latest version docstrings of the method otherwise use the one bellow


## 1. When asked about creation a `TimeSeries` the following methods are obligatory

### __init__

#### best practices

* Always decoreate the method with @TimeSerie._post_init_routines
* Only use allowed types in the signature.
* When you can to build a TimeSerie dependecy to another TimeSerie you must initialize is inside of this method for example.
* It is best practice when a TimeSerie task makes a calculation on assets and one of the argument is of type ModelList like asset_list to include asset_list in local_kwargs_to_ignore
```python
class TimeSerie1:
    def __init__(self,arg1:str,*args,**kwargs):
        
        self.arg1=arg1
        super().__init__(*args,**kwargs)
class TimeSerie2:
    def __init__(self,arg1:str,*args,**kwargs):
        
        self.time_serie1=TimeSerie1(arg1=arg1,*args,**kwargs)
        super().__init__(*args,**kwargs)

```

#### code reference

```python

def __init__(
        self,
        init_meta: Optional[TimeSerieInitMeta] = None,
        build_meta_data: Optional[dict] = None,
        local_kwargs_to_ignore: Optional[List[str]] = None,
        *args,
        **kwargs
):
    """
    Create a TimeSerie instance and provision its storage in the Main Sequence Data Engine.

    This initializer prepares all metadata and configuration for a new time series, then:
      1. Computes a unique table identifier by hashing all arguments **except**
         `init_meta`, `build_meta_data`, and `local_kwargs_to_ignore`.
      2. Computes a separate `local_hash_id` (and corresponding LocalTimeSerie) by
         hashing the same inputs, but omitting any keys listed in `local_kwargs_to_ignore`.
      3. Applies any `@TimeSerie._post_init_routines` decorators after setup.

    Argument types must be one of:
      - `str`, `int`, `list`,
      - Pydantic model instances or lists thereof.

    You may override the class constant `OFFSET_START` to define the earliest
    timestamp into which new data will be inserted.

    Parameters
    ----------
    init_meta : TimeSerieInitMeta, optional
        Initial metadata for this time series (e.g. name, description).
    build_meta_data : dict, optional
        Metadata recorded during the build process (e.g. version, source tags).
    local_kwargs_to_ignore : list of str, optional
        Names of keyword args that should be excluded when computing `local_hash_id`.
    *args
        Positional arguments used to further parameterize the series.
    **kwargs
        Keyword arguments used to further parameterize the series.
    """

```

### update

#### best practices

* Always check that the update method covers the cases when the time serie is updated and when it is not. in the case of tables with two indices time_index, unique_identifier
update_statistics.update_statistics will have an overriden value corresponding to `OFFSET_START` so there is no need to use two cases
only in the special case that the `TimeSerie` requires the last observation value for example with `O(1)` In this case you should use the
method `.self.get_df_between_dates` 

* The method `TimeSerie.get_df_between_dates` should be the only method used to get data from another time series in cases
when working with double index time_serie,unique_indetifier is recommended to use `unique_identifier_range_map`

* Be sure to return only dataframes as specified in the docstrings
* This method should never be called for any process is part of the general mainsequence engine just need to be implemented



```python

class DateInfo(TypedDict, total=False):
    start_date: Optional[datetime.datetime]
    start_date_operand: Optional[str]
    end_date: Optional[datetime.datetime]
    end_date_operand: Optional[str]

UniqueIdentifierRangeMap = Dict[str, DateInfo]

class TimeSerie:
            def get_df_between_dates(
            self,
            start_date: Union[datetime.datetime, None] = None,
            end_date: Union[datetime.datetime, None] = None,
            unique_identifier_list: Union[None, list] = None,
            great_or_equal: bool = True,
            less_or_equal: bool = True,
            unique_identifier_range_map: Optional[UniqueIdentifierRangeMap] = None,
    ) -> pd.DataFrame:
        """
        Retrieve rows from this TimeSerie whose `time_index` (and optional `unique_identifier`) fall within the specified date ranges.

        **Note:** If `unique_identifier_range_map` is provided, **all** other filters
        (`start_date`, `end_date`, `unique_identifier_list`, `great_or_equal`, `less_or_equal`)
        are ignored, and only the per-identifier ranges in `unique_identifier_range_map` apply.

        Filtering logic (when `unique_identifier_range_map` is None):
          - If `start_date` is provided, include rows where
            `time_index > start_date` (if `great_or_equal=False`)
            or `time_index >= start_date` (if `great_or_equal=True`).
          - If `end_date` is provided, include rows where
            `time_index < end_date` (if `less_or_equal=False`)
            or `time_index <= end_date` (if `less_or_equal=True`).
          - If `unique_identifier_list` is provided, only include rows whose
            `unique_identifier` is in that list.

        Filtering logic (when `unique_identifier_range_map` is provided):
          - For each `unique_identifier`, apply its own `start_date`/`end_date`
            filters using the specified operands (`">"`, `">="`, `"<"`, `"<="`):
            {
              <uid>: {
                "start_date": datetime,
                "start_date_operand": ">=" or ">",
                "end_date": datetime,
                "end_date_operand": "<=" or "<"
              },
              ...
            }

        Parameters
        ----------
        start_date : datetime.datetime or None
            Global lower bound for `time_index`. Ignored if `unique_identifier_range_map` is provided.
        end_date : datetime.datetime or None
            Global upper bound for `time_index`. Ignored if `unique_identifier_range_map` is provided.
        unique_identifier_list : list or None
            If provided, only include rows matching these IDs. Ignored if `unique_identifier_range_map` is provided.
        great_or_equal : bool, default True
            If True, use `>=` when filtering by `start_date`; otherwise use `>`. Ignored if `unique_identifier_range_map` is provided.
        less_or_equal : bool, default True
            If True, use `<=` when filtering by `end_date`; otherwise use `<`. Ignored if `unique_identifier_range_map` is provided.
        unique_identifier_range_map : UniqueIdentifierRangeMap or None
            Mapping of specific `unique_identifier` keys to their own sub-filters. When provided, this is the sole filter applied.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing rows that satisfy the combined time and identifier filters.
        """
        

```
#### code reference
```python

    def update(self, update_statistics: DataUpdates) -> pd.DataFrame:
        """
        Fetch and ingest only the new rows for this TimeSerie based on prior update checkpoints.

        DataUpdates provides the last-ingested positions:
          - For a single-index series (time_index only), `update_statistics.max_time` is either:
              - None: no prior data—fetch all available rows.
              - a datetime: fetch rows where `time_index > max_time`.
          - For a dual-index series (time_index, unique_identifier), `update_statistics.max_time_per_id` is either:
              - None: single-index behavior applies.
              - dict[str, datetime]: for each `unique_identifier` (matching `Asset.unique_identifier`), fetch rows where 
                `time_index > max_time_per_id[unique_identifier]`.

        Requirements:
          - `time_index` **must** be a `datetime.datetime` instance with UTC timezone.
          - Column names **must** be all lowercase.
          - No column values may be Python `datetime` objects; if date/time storage is needed, convert to integer
            timestamps (e.g., UNIX epoch in seconds or milliseconds).

        After retrieving the incremental rows, this method inserts or upserts them into the Main Sequence Data Engine.

        Parameters
        ----------
        update_statistics : DataUpdates
            Object capturing the previous update state. Must expose:
              - `max_time` (datetime | None)
              - `max_time_per_id` (dict[str, datetime] | None)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the newly added or updated records.
        """
        raise NotImplementedError

```




### _get_column_metadata

While this method is not obligatory is also a best practice to have as it helps the Main Sequence Data Engine

```python
    def _get_column_metadata(self)->list[ColumnMetaData]:
        """
        This Method should return a list for ColumnMetaData to add extra context to each time series
        Examples:
            from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [ColumnMetaData(column_name="instrument",
                                          dtype="str",
                                          label="Instrument",
                                          description=(
                                              "Unique identifier provided by Valmer; it’s a composition of the "
                                              "columns `tv_emisora_serie`, and is also used as a ticker for custom "
                                              "assets in Valmer."
                                          )
                                          ),
                            ColumnMetaData(column_name="currency",
                                           dtype="str",
                                           label="Currency",
                                           description=(
                                               "Corresponds to  code for curries be aware this may not match Figi Currency assets"
                                           )
                                           ),
                            
                            ]
            
        
        Returns:

        """
   


```




## 2. When asked about creation a `TimeSeries` sometimes the following method is necessary

### _get_get_asset_list

#### Best Practices
* When a time_serie has a double index : time_index, unique_identifier and asset_list is not set in the constructor
this method is oblogatory. One example of this case is when we are building an update given an asset_category.unique_identifier
* Having asset categoies in the constructor rathern than asset list is also a best practice as it keep clean the constructor confitguration
* this method should always return a list of  mainsequence.client.Asset

#### Code Reference

```python
    def _get_asset_list(self) -> Optional[List["Asset"]]:

        """
        Provide the list of assets that this TimeSerie should include when updating.

        By default, this method returns `self.asset_list` if defined.
        Subclasses _must_ override this method when no `asset_list` attribute was set
        during initialization, to supply a dynamic list of assets for update_statistics.

        Use Case:
          - For category-based series, return all Asset unique_identifiers in a given category
            (e.g., `AssetCategory(unique_identifier="investable_assets")`), so that only those
            assets are updated in this TimeSerie.

        Returns
        -------
        list or None
            - A list of asset unique_identifiers to include in the update.
            - `None` if no filtering by asset is required (update all assets by default).
        """
        if hasattr(self, "asset_list"):
            return self.asset_list

        return None
```

## 3. When asked about creation a `TimeSeries` The following method should be implemented if we want to make it a MarketTimeSerie

When this method is implemeted a secondary object called `MarketTimeSerie` is created