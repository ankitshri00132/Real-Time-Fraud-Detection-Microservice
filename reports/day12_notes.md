# Day 12 Summary

- Logging was implemented to track API requests, model predictions, and system errors. A centralized logging system was created using Python’s logging module, storing logs in both files and console output. Logs capture important events such as prediction start, prediction results, and failures. This improves debugging, traceability, and production reliability.

- A monitoring system was built to track key metrics such as total requests, predictions, fraud detections, and errors. A dedicated /metrics endpoint was created to expose system health statistics. This enables real-time tracking of model usage and performance.
