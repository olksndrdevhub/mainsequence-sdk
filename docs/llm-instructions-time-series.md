from mainsequence.tdag import APITimeSerie

## DataNode Implementation Guidelines  
*Main Sequence Platform*

These guidelines cover the essential configuration and best-practice patterns for extending the `DataNode` class on the Main Sequence platform.

> **Tip — docstrings first**  
> Always check the most recent `DataNode` source for updated docstrings.  
> If they differ from this document, treat the source code as the single source of truth.


> **Tip — docstrings first**  
> import mainsequence.client as ms_client for cleare readability 


---

### 1&nbsp;&nbsp;Required methods

#### **`__init__`**

**Best practices**

* Decorate the constructor with `@DataNode._post_init_routines`.
* Accept only these argument types: `str`, `int`, `list`, Pydantic model, or a list of Pydantic models.
* Instantiate any dependent `DataNode` objects inside the constructor.
* If the series receives a `ModelList` argument such as `asset_list`, add its name to `local_kwargs_to_ignore`.
* `TimeSeries` can have direct dependencies which are initialized directly in the construction with its imported class or `APITimeSeries` which should be also initialized in the construction but are initialized only with `local_hash_id` and `data_source_id`
* You may override the class constant `OFFSET_START` to set the earliest
   permitted timestamp.

##### Example of A `DataNode` direct dependency
```python
from mainsequence.tdag import DataNode
class PriceSerie(DataNode):
    @DataNode._post_init_routines
    def __init__(self, arg1: str, *args, **kwargs):
        self.arg1 = arg1
        super().__init__(*args, **kwargs)

class ReturnSerie(DataNode):
    @DataNode._post_init_routines
    def __init__(self, arg1: str, *args, **kwargs):
        self.price_serie = PriceSerie(arg1=arg1, *args, **kwargs)
        super().__init__(*args, **kwargs)
```
##### Example of A `APITimeSerie` direct dependency
```python
from mainsequence.tdag import APITimeSerie
class ReturnSerie(DataNode):
    @DataNode._post_init_routines
    def __init__(self, arg1: str, *args, **kwargs):
        self.price_serie = APITimeSerie(data_source_id=1,local_hash_id="local_hash_id_ghjkdf8347y5342")
        super().__init__(*args, **kwargs)
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
    Create a DataNode instance and provision its storage in the Main Sequence Data Engine.

    This initializer prepares all metadata and configuration for a new time series, then:
      1. Computes a unique table identifier by hashing all arguments **except**
         `init_meta`, `build_meta_data`, and `local_kwargs_to_ignore`.
      2. Computes a separate `local_hash_id` (and corresponding LocalTimeSerie) by
         hashing the same inputs, but omitting any keys listed in `local_kwargs_to_ignore`.
      3. Applies any `@DataNode._post_init_routines` decorators after setup.

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

**Best practices**


* Support both first-time ingestions and incremental updates.
* Dual-index tables ( `time_index`, `unique_identifier` ) do **not** require two
  separate code paths as `update_statistics.unique_identifier` already references
  `OFFSET_START`. for single-index get the last update from `update_statistics.get_max_latest_value`
* Use `self.get_df_between_dates(…)` to read from other time series;  
  for dual-index reads, prefer `unique_identifier_range_map`.
* Return a DataFrame and nothing else.
* The engine never calls `update()` directly, but every subclass must implement it.


**Functional contract (summary)**

* For single-index series: fetch rows where `time_index` is greater than the
  stored `max_time`.
* For dual-index series: fetch rows where `time_index` is greater than the
  per-identifier checkpoint in `max_time_per_id`.
* All ingested `time_index` values must be UTC-aware `datetime`.
* Column names must be lowercase; value columns must not contain raw
  `datetime` objects (store integers such as Unix epochs instead).




```python

class DateInfo(TypedDict, total=False):
    start_date: Optional[datetime.datetime]
    start_date_operand: Optional[str]
    end_date: Optional[datetime.datetime]
    end_date_operand: Optional[str]

UniqueIdentifierRangeMap = Dict[str, DateInfo]

class DataNode:
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
        Retrieve rows from this DataNode whose `time_index` (and optional `unique_identifier`) fall within the specified date ranges.

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
        Fetch and ingest only the new rows for this DataNode based on prior update checkpoints.

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




### `get_column_metadata() → list[ColumnMetaData]`  *(recommended)*

While this method is not obligatory is also a best practice to have as it helps the Main Sequence Data Engine.
*Return a list that documents each column’s name, data type, label, and description.*  
This enriches downstream applications and is strongly encouraged for every new series.

```python
    def get_column_metadata(self)->list[ColumnMetaData]:
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




## 2  Conditionally Required Methods

### `get_asset_list() → Optional[List[Asset]]`

Implement this method **only** when both conditions hold:

1. The series uses the dual index `(time_index, unique_identifier)`.
2. No `asset_list` was provided in the constructor.


**Guidelines**

* Prefer passing an asset *category* in `__init__`; compute the concrete list here.
* If `self.asset_list` exists, simply return it; otherwise assemble the list
  dynamically (for example, from an `AssetCategory`).


#### Code Reference

```python
    def get_asset_list(self) -> Optional[List["Asset"]]:

        """
        Provide the list of assets that this DataNode should include when updating.

        By default, this method returns `self.asset_list` if defined.
        Subclasses _must_ override this method when no `asset_list` attribute was set
        during initialization, to supply a dynamic list of assets for update_statistics.

        Use Case:
          - For category-based series, return all Asset unique_identifiers in a given category
            (e.g., `AssetCategory(unique_identifier="investable_assets")`), so that only those
            assets are updated in this DataNode.

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

## 3  Creating a `MarketTimeSerie`

A **`MarketTimeSerie`** is a registry entry that exposes a `DataNode` in the
Markets catalogue with a clean, user-friendly identifier.  
Create (or update) one whenever **either** of the following is true:

* The `DataNode` has an `asset_list` and should be easy to discover in the
  Markets UI / API.
* The series is macro-level (factor, index, benchmark, etc.) and benefits from a
  human-readable name.

The preferred implementation site is the hook  
`_run_post_update_routines(self, error_on_last_update: bool, update_statistics: DataUpdates)`,  
which executes immediately after each successful `update()`.

### Generic pattern

```python
import mainsequence.client as ms_client
from mainsequence.tdag import DataNode
class YourTimeSerie(DataNode):
    ...

    def _run_post_update_routines(
        self,
        error_on_last_update: bool,
        update_statistics: DataUpdates,
    ) -> None:
        """
        Register—or refresh—a MarketsTimeSeriesDetails record so this
        DataNode is discoverable in the Markets platform.
        """
        if error_on_last_update:      # Skip registration if the update failed
            return

        UNIQUE_ID    = "<your_unique_snake_case_id>"          # e.g. "sp500_total_return"
        FREQUENCY_ID = ms_client.DataFrequency.one_d          # pick the correct enum
        DESCRIPTION  = (
            "Plain-English description of what the data represents and "
            "its provenance (source, methodology, units)."
        )

        try:
            mts = ms_client.MarketsTimeSeriesDetails.get(unique_identifier=UNIQUE_ID)

            # Re-link if this DataNode was rebuilt and now has a new LocalTimeSerie
            if mts.related_local_time_serie.id != self.local_time_serie.id:
                mts.patch(related_local_time_serie__id=self.local_time_serie.id)

        except ms_client.DoesNotExist:
            ms_client.MarketsTimeSeriesDetails.update_or_create(
                unique_identifier            = UNIQUE_ID,
                related_local_time_serie__id = self.local_time_serie.id,
                data_frequency_id            = FREQUENCY_ID,
                description                  = DESCRIPTION,
            )
```


## 4  Data Retrieval Helper

Always use `get_df_between_dates(…)` for inter-series reads.  
For dual-index filters, supply `unique_identifier_range_map` to achieve
per-identifier date windows.

## 5  General Conventions

* Column names: **lowercase only**.  
* `time_index`: timezone-aware `datetime` in UTC.  
* Value-column datetimes: store as integers (e.g. Unix epoch).  
* Keep constructors minimal; move complex wiring into separate factories.  
* Provide thorough unit tests for constructor logic, `update`, and any
  transformation utilities.

---