# Spark Configuration Tool

https://spark-config-calculator.streamlit.app/

## Overview

The Spark Configuration Tool is a Streamlit-based application designed to assist users in optimizing Apache Spark configurations. It allows users to input various parameters related to cluster, node, and executor configurations, providing recommended Spark configurations based on those inputs.

## Features

- **Cluster Configuration:** Configure the number of workers and choose between Dominant Resource Calculator and Default Resource Calculator for the Capacity Scheduler.
  
- **Node Configuration:** Specify total RAM per node, cores per node, and percent RAM for the OS. It also offers options to reserve a core for the OS daemon and oversubscribe CPU.

- **Executor Configuration:** Adjust settings such as executor cores, memory overhead, memory fraction, storage fraction, and off-heap memory.

- **Metrics and Breakdowns:** The tool provides metrics on memory and CPU utilization, along with breakdowns of on-heap memory usage.

- **Revised Configurations:** Generate revised Spark configurations based on user inputs and dynamically adjust memory and core allocations.

## Authored By
Jerome Rajan
www.jeromerajan.com
