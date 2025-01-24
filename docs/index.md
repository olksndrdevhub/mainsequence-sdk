# Introduction to the Main Sequence Python SDK

The **Main Sequence Python SDK** is a powerful set of client libraries designed to facilitate interaction with Main Sequence systems using Python. The SDK covers three main components:

1. **TDAG:** A time-directed acyclic graph system for managing complex, time-sensitive data pipelines.
2. **TDAG Client:** A set of client methods to interact with the TDAG backend.
3. **VAM Client:** A set of client methods to interact with the VAM backend.

Several workflows demonstrating the usage of the SDK can be found in our [example repository](https://github.com/mainsequence/sdk-examples).

---

## What is TDAG?

At its core, **TDAG** leverages the power of **DAGs** (Directed Acyclic Graphs). A DAG is a graph with nodes connected by edges, where the edges have a direction, and no cycles (loops) exist. In simpler terms, this structure allows data to flow in one direction without any feedback loops, which is essential for building reliable and predictable data pipelines. The "time-directed" aspect in TDAG makes it ideal for handling time-sensitive operations, ensuring that tasks occur in the correct sequence.

### Why TDAG?

With TDAG, you can automatically create **time-based data pipelines** that handle complex dependencies. The library provides features such as automatic hashing, seamless scheduling integration, and a structured approach that enhances the reliability and scalability of your pipelines.

### Key Features of TDAG

- **Automated Time-Based Pipelines:** TDAG allows you to easily build data pipelines where tasks are executed in time order, respecting dependencies.
- **Built-in Scheduling and Hashing:** Pipelines are automatically hashed and scheduled for efficient execution.
- **Scalable & Robust:** Whether you're working on small datasets or massive time-series data flows, TDAG scales to meet your needs while ensuring the entire process is fault-tolerant and robust.

### Use Case: Investment Strategies and Beyond

One of TDAG's main use cases is transforming raw financial data into actionable insights, such as investment strategy predictions or portfolio weights. TDAG simplifies the process of managing complex time-based operations in **financial modeling**, helping you move from data to decisions effortlessly.

However, **TDAG** is not just limited to finance! It's perfect for **any application requiring time-sensitive data pipelines**, particularly in **live and online modes** where real-time decision-making is crucial. For example, TDAG can be used in **online training of machine learning models**, where time-based data flow and immediate processing are essential for model accuracy and performance.

---

## TDAG Client

The **TDAG Client** provides the necessary methods to interact with the TDAG backend. It enables users to submit, track, and manage their TDAG-based workflows programmatically, ensuring smooth execution and monitoring of time-sensitive operations.

## VAM Client

The **VAM Client** offers methods to interact with the VAM backend,
facilitating the retrieval and processing of valuation and analytics data. 
This allows seamless integration of asset valuation within the broader financial modeling ecosystem.

---

## Explore More

To get started with the SDK and explore practical examples, check out our:

- 📖 **[TDAG Tutorial](tdag/tutorial/getting_started/Introduction_part1.md)** – Start with the TDAG Tutorial if you are looking to build customized portfolios.
- 📂 **[Example Repository](https://github.com/mainsequence/sdk-examples)** – Find practical examples and workflows to kickstart your journey with the SDK.

---

## Sections of this Documentation

- [Getting Started](getting_started.md)
- [Installation](installation.md)
- [Key Concepts](concepts.md)
- [Tutorials](tutorials.md)
- [Usage Examples](examples.md)
- [API Reference](reference.md)
- [FAQ](faq.md)
- [Contributing](contributing.md)
- [Changelog](changelog.md)

Explore the sidebar for more detailed topics and guides.


