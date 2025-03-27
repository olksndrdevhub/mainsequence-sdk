# Time Series Fundamentals in TDAG

TDAG's core revolves around the concept of time series, encapsulated in the powerful `TimeSerie` class. Each `TimeSerie` object manages the process of updating data, reflecting the most current available information. It interacts seamlessly with databases and maintains a robust internal state, supporting efficient data pipelines.

## Understanding the Update Process

In TDAG, updating involves:

- **Updating DataRepositories**: Stores the generated data from the update process.
- **Updating ORM**: Manages the internal state of the data and the pipeline.

The following diagram illustrates these interactions:

```mermaid
flowchart TD
    subgraph TDAG_System[TDAG Framework]
         TimeSerieConstructor["TimeSerie.__init__(*args, **kwargs)"] -->|Defines| TimeSerie["TimeSerie.update(latest_value)"]
    end

    subgraph DataRepositories["Data Repositories"]
        DB["TimeScaleDB"]
        DataLake["DataLake"]
    end

    subgraph ORM["TDAG ORM"]
        LocalTimeSerie["LocalTimeSerie (local_hash_id)"]  -->|View of Table| DynamicTable["DynamicTable (hash_id)"]
    end

    TimeSerie -->|Updates| DataRepositories
    TimeSerie -->|Updates| ORM
     ORM -->|Manages state of| DataRepositories
    ORM -->|Updates| DynamicTable
    ORM -->|Updates| LocalTimeSerie
```

## Initializing a TimeSerie

The constructor (`__init__`) defines the initial state and configuration:

```python
def __init__(self, init_meta=None,
             build_meta_data: Union[dict, None] = None,
             local_kwargs_to_ignore: Union[List[str], None] = None,
             data_configuration_path: Union[str, None] = None,
             *args, **kwargs):
    ...
```

### Hashing Mechanism

The constructor arguments create two essential hashes that facilitate efficient management of data and updates:

- **`hash_id`**: Used to uniquely identify data repositories linked to a specific `TimeSerie`. This ensures different configurations or datasets are appropriately separated or merged based on their content rather than their names alone.

- **`local_hash_id`**: Used to uniquely identify the specific update processes. It enables TDAG to recognize distinct update routines and manage their internal state independently, crucial for parallel updates or workflows that reuse identical data structures with different update logic.

### Special Constructor Arguments

Some arguments are explicitly excluded from hashing:

- **`init_meta`**: Arbitrary metadata used during initialization for convenience and clarity.
- **`build_meta_data`**: Metadata recoverable anytime and editable from interfaces; useful for dynamic or interactive data handling.
- **`local_kwargs_to_ignore`**: Arguments excluded from the `hash_id` calculation but included in `local_hash_id`, allowing flexibility in differentiating between datasets and update processes.

### Decorator Usage

Always decorate the constructor to ensure proper integration with TDAG:

```python
from mainsequence.tdag import TimeSerie

class NewTimeSeries(TimeSerie):
    @TimeSerie._post_init_routines
    def __init__(self):
        ...
```

### Managing Dependencies with Introspection

TDAG simplifies dependency management by automatically detecting dependencies through introspection. Rather than manually managing complex dependency trees, developers only need to explicitly declare direct dependencies as class attributes. TDAG then builds the full dependency graph internally.

Example:

```python
class NewTimeSeries(TimeSerie):
    @TimeSerie._post_init_routines
    def __init__(self, asset_symbols: List[str], *args, **kwargs):
        # Explicitly declare direct dependency
        self.prices_time_serie = PricesTimeSerie(asset_symbols=asset_symbols)
```

TDAG automatically understands that `NewTimeSeries` depends on `PricesTimeSerie` and manages updates accordingly.

### State Persistence with Pickles

TDAG pickles each `TimeSerie` after its first initialization, significantly reducing load times in future updates. The pickle state is automatically updated when the underlying code changes, ensuring consistency and efficiency.

## Updating a TimeSerie

The `update` method performs all the necessary logic to fetch, calculate, and store new data points in the series. It uses a parameter called `latest_value`, representing the most recent timestamp from previous updates. If `latest_value` is `None`, the series has never been updated successfully before. Otherwise, it continues from the given point.

Example:

```python
def update(self, latest_value: Union[None, datetime.datetime], *args, **kwargs) -> pd.DataFrame:
    # Perform update logic based on latest_value
    new_data = self.fetch_new_data_since(latest_value)
    processed_data = self.calculate_metrics(new_data)
    return processed_data
```

Returned DataFrame requirements:

- **Unidimensional index**: `DatetimeIndex` in `pytz.utc`.
- **Multidimensional index**: three dimensions: `time_index`, `asset_symbol`, and `execution_venue_symbol`.

## Essential Helper Methods

### Retrieving Data Between Dates

The `get_df_between_dates` method fetches data from a specified date range:

```python
def get_df_between_dates(self, start_date: Union[datetime.datetime, None] = None,
                         end_date: Union[datetime.datetime, None] = None,
                         unique_identifier_list: Union[None, list] = None,
                         great_or_equal=True, less_or_equal=True,
                         unique_identifier_range_map: Optional[UniqueIdentifierRangeMap] = None
                         ) -> pd.DataFrame:
```

This method efficiently retrieves data within a specific range, accommodating various filtering scenarios by asset or time intervals.

### Getting Last Observation

```python
def get_last_observation(self, asset_symbols: Union[None, list] = None):
```

Returns the most recent observation in the series.

## WrapperTimeSerie Class

`WrapperTimeSerie` allows collective management of multiple `TimeSerie` instances, facilitating scalable and efficient updates:

```python
class WrapperTimeSerie(TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self, time_series_dict: Dict[str, TimeSerie], *args, **kwargs):
```

TDAG’s TimeSeries object simplifies managing complex data dependencies, ensuring efficiency, consistency, and maintainability in your data pipelines.
