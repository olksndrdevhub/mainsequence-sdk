# TDAG configuration

When tdag is run for the first time it will create the following tree structure

```
├── config.yml
├── data
│   ├── logs
│   │   ├── schedulers
│   │   └── time_series
│   └── time_series_data
│       └── pickled_ts
├── .env
├── ray
└── temp
```
## config.yml

This file contains tdag main configuration that does not include sensitive data