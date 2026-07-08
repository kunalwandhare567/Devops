Log Feature Engineering Workflow (Algorithm Independent)
This document describes the complete workflow for converting raw log data into a structured feature matrix. It focuses only on data preparation and feature engineering and does not assume any specific machine learning algorithm.
Step 1: Data Collection
Collect raw log data from all 14 failure modes (Memory Leak, Bad Deploy, CPU Saturation, etc.). Combine the logs into a single dataset for processing.
Step 2: Data Cleaning & Preprocessing
Remove duplicate logs, handle missing values, standardize timestamps and log formats, and discard corrupted records.
Step 3: Train-Test Split (80:20)
Split the dataset into 80% training data and 20% testing data. Use stratified sampling so that all 14 failure modes are represented in both datasets.
Step 4: Log Parsing
Parse each log entry and extract structured fields such as Timestamp, Service Name, Log Level, Exception Type, and Log Message.
Step 5: Log Template Extraction (Drain3)
Use the Drain3 library to group similar log messages into templates by replacing variable values with placeholders. This standardizes logs and identifies repeated patterns.
Step 6: Feature Engineering
Generate numerical and categorical features from the parsed logs and Drain3 templates.
Step 7: Feature Matrix Creation
Create X (input features) and Y (target labels). X contains engineered features, while Y contains the corresponding failure mode labels.
Feature Engineering Output
Feature	How it is Created	Purpose
log_count	Count total log entries	Measures system activity
log_max_severity	Find highest log severity	Measures issue severity
log_critical_count	Count CRITICAL logs	Detects severe incidents
error_count	Count ERROR logs	Measures application failures
warning_count	Count WARNING logs	Provides early warning
log_has_exception	Check whether an exception exists	Detects runtime failures
log_has_novel_template	Compare Drain3 template with known templates	Detects unseen log patterns
Feature Matrix
X (Input Features):
log_count, log_max_severity, log_critical_count, error_count, warning_count, log_has_exception, log_has_novel_template

Y (Target Label):
Failure Mode (Memory Leak, Bad Deploy, CPU Saturation, Database Slowdown, etc.).
Overall Workflow
Raw Log Data
      ↓
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Train-Test Split (80:20)
      ↓
Log Parsing
      ↓
Drain3 Log Template Extraction
      ↓
Feature Engineering
      ↓
Feature Matrix
      ├── X (Input Features)
      └── Y (Failure Mode Labels)
