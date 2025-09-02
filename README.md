IoTStream – Real-Time Sensor Monitoring and Anomaly Detection

I developed IoTStream, a Python-based IoT simulation platform designed to generate, process, and visualize sensor data in real-time. The goal of the project was to create a complete end-to-end system that mimics the behavior of real-world IoT networks, focusing on both data analysis and actionable monitoring.

The system simulates multiple virtual sensors that continuously capture temperature and humidity readings. These data streams are then processed through a series of transformations, including rolling averages to smooth fluctuations and difference calculations to highlight sudden changes. By applying anomaly detection at both the individual sensor level and across combined sensor metrics, IoTStream is able to identify irregular patterns that may indicate system faults, environmental changes, or unexpected behavior.

A unique feature of the system is its alerting mechanism, where consecutive anomalies automatically trigger notifications to simulate real-world IoT responses. Additionally, IoTStream integrates with SQLite, enabling advanced querying, filtering, and aggregation of sensor data. This allows for efficient storage, retrieval, and deeper exploration of sensor metrics over time.

To make insights more accessible, the project includes comprehensive visualizations using matplotlib and seaborn. These visual dashboards highlight trends, correlations, and anomaly points for each sensor, helping users quickly interpret system performance and identify emerging issues.

Overall, this project demonstrates strong proficiency in Python programming, pandas for data manipulation, SQLite for database management, and matplotlib/seaborn for visualization. It also highlights my ability to design and implement a full-stack IoT data handling pipeline, incorporating real-time monitoring, anomaly detection, database integration, and clear visual communication of results.
