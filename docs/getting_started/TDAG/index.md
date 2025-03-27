# Welcome to TDAG

TDAG is a cutting-edge, graph-based workflow orchestration tool specifically designed to simplify the creation, management, and scaling of time-series data pipelines. Unlike general-purpose orchestration tools such as Airflow or Luigi, TDAG is purpose-built with the performance, scalability, and flexibility needed for modern asset management and real-time data processing.
## Why Another Workflow Orchestration Tool?

While orchestration tools like Airflow and Luigi are widely used, they weren't originally built to handle the demanding performance and resource constraints required by modern asset management workflows. TDAG addresses these limitations by providing native support for resource management (CPU/GPU), seamless integration with databases, and built-in time-series logic that reduces boilerplate code and development overhead.
## Key Features of TDAG:

- **Native Time-Series Support:** Built specifically for investment management and real-time data operations, TDAG provides first-class concepts like `TimeSerie` and `update_statistics`, streamlining pipeline creation.

- **Integrated Resource Management:** TDAG enforces CPU and GPU resource constraints directly within workflow definitions, simplifying resource allocation without requiring external schedulers.

- **Built-in Data Layer:** TDAG offers integrated data-handling features (`update_statistics` and `get_last_observation()`), eliminating the need for extensive custom database interaction code.

- **Automated Dependency Management:** TDAG automatically manages complex task dependencies, significantly reducing manual setup and ensuring efficient execution.

- **High Performance and Scalability:** Optimized specifically for high-throughput environments, TDAG excels at scaling seamlessly with demanding time-series tasks.


## Example Use Case: Simple Time Series Pipeline

 Imagine a scenario where each task sequentially processes data:
  
  - **Task A:** Inserts an initial value.
  - **Task B:** Takes the output from Task A, adds 5, and inserts the result.
  - **Task C:** Takes the output from Task B, adds another 5, and inserts the final result.
  
  Each of these tasks requires defined resources (e.g., 2 GPUs and 10 CPUs).
  
  While Luigi and Airflow can achieve this, both require significant extra setup, particularly for GPU management 
  and seamless database integration. 
  TDAG simplifies this process dramatically, embedding these features directly into workflow definitions, as demonstrated below:

```python
from mainsequence.tdag import TimeSerie
from mainsequence.tdag_client.models import DataUpdates
import pandas as pd
import datetime

class TS1(TimeSerie):
    REQUIRED_CPUS = 10
    REQUIRED_GPUS = 2

    def update(self, update_statistics: DataUpdates):
        data = pd.DataFrame(index=[datetime.datetime.now()], data=[100])
        return data

class TS2(TimeSerie):
    REQUIRED_CPUS = 10
    REQUIRED_GPUS = 2

    def __init__(self):
        self.ts_dep = TS1()

    def update(self, update_statistics: DataUpdates):
        last_val = self.ts_dep.get_last_observation().values()[0]
        data = pd.DataFrame(index=[datetime.datetime.now()], data=[last_val + 5])
        return data

class TS3(TimeSerie):
    REQUIRED_CPUS = 10
    REQUIRED_GPUS = 2

    def __init__(self):
        self.ts_dep = TS2()

    def update(self, update_statistics: DataUpdates):
        last_val = self.ts_dep.get_last_observation().values()[0]
        data = pd.DataFrame(index=[datetime.datetime.now()], data=[last_val + 5])
        return data
```

## TDAG vs. Luigi vs. Airflow

| Feature                 | Luigi                               | Airflow                          | TDAG (Main Sequence)                        |
|-------------------------|-------------------------------------|----------------------------------|---------------------------------------------|
| **Database**            | SQLite (local file)                 | SQLite (local file)              | TimeScale or any supported database         |
| **Resources**           | Defined in Task (manual config)     | Defined explicitly in Executors  | Declared directly in classes (`REQUIRED_GPUS`) |
| **DAG Definition**      | Python Classes with `requires()`    | Python Operators with `>>`       | Python Classes with automatic dependencies  |
| **Scheduling**          | External scheduler required         | Built-in scheduler (UI)          | Integrated scheduler in Main Sequence platform |
| **Execution & Monitoring** | CLI                               | Web Interface or CLI             | Main Sequence UI or CLI                     |


## Real-World Applications

- Financial modeling and investment strategies
- Real-time machine learning workflows
- Online training and predictive analytics

TDAG ensures that your data pipelines are robust, scalable, and easy to maintain, enabling your teams to focus on generating value rather than managing infrastructure.

