from dotenv import load_dotenv
load_dotenv()

import streamlit as st
st.set_page_config(page_title="AI-buddy", page_icon="logo2.png", layout="centered")

import os
import mysql.connector
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import re
import time
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import speech_recognition as sr
from deep_translator import GoogleTranslator
import json
import numpy as np
import tempfile
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import bcrypt
import base64
from streamlit_cookies_manager import EncryptedCookieManager
import streamlit.components.v1 as components 
import asyncio
import edge_tts
import plotly.io as pio
from flask import Flask,Response, request, send_file,jsonify
import threading
import uuid
from functools import wraps
import logging
import urllib
from flask_cors import CORS
from waitress import serve
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from difflib import SequenceMatcher

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# this is for copy button to work, unique id is required for each message.
if "global_message_id" not in st.session_state:  
    st.session_state.unique_copy_id = 1

# Initialize the NTP client
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
# Define the comprehensive prompt
SYSTEM_PROMPT = """
You are an expert in converting English questions to MYSQL queries!. The MYSQL database has the following tables:

1. **machine_details**: with columns:
   - `machine_id` (int, AI, PK)
   - `connection_timestamp` (datetime)
   - `machine_name` (varchar(50))
   - `ip_address` (varchar(50))
   - `office_id` (int)
   - `factory_id` (varchar(200))
   - `shop_floor_id` (int)
   - `production_line_id` (int)
   - `minder_ip_address` (varchar(255))

2. **alarm_frequency_log**: with columns:
   - `alarm_frequency_id` (int, AI, PK)
   - `alarm_number` (int)
   - `alarm_frequency` (int)
   - `ip_address` (varchar(50))
   - `alarm_set_timestamp` (datetime)
   
3.**alarm_dictionary**: with columns:
   - `alarm_description_id` (int, AI, PK)
   - `alarm_number` (int, FK)
   - `alarm_description` (varchar(1024))

4.**machine_parameters**: with columns:
     - `parameter_id` (int, AI, PK)
     - `timestamp` (datetime)
     - `ip_address` (varchar(50))
     - `machine_name` (varchar(50))
     - `machine_number` (varchar(16))
     - `machine_configuration` (varchar(50))
     - `mold_id` (varchar(16))
     - `part_number` (varchar(16))
     - `material_name` (varchar(16))
     - `machine_mode` (int)
     - `cavity_count` (int)
     - `pump_status` (int)
     - `heater_status` (int)
     - `shot_weight` (float)
     - `actual_good_parts` (int)
     - `actual_rejected_parts` (int)
     - `actual_good_shots` (int)
     - `actual_total_shots` (int)
     - `bin_actual_good_parts` (int)
     - `bin_actual_rejected_parts` (int)
     - `bin_actual_good_shots` (int)
     - `bin_actual_total_shots` (int)
     - `good_parts_target` (int)
     - `energy_consumption` (int)
     - `ideal_cycle_time_seconds` (float)

5. **downtime_logs**: with columns:
     - `downtime_id` (int, AI, PK)
     - `ip_address` (varchar(255))
     - `from_time` (datetime)
     - `to_time` (datetime)
     - `downtime_interval` (time)
     - `downtime_reason_id` (int)
     - `downtime_remarks` (varchar(50))
     - `downtime_approval` (int)
     - `downtime_category` (varchar(45))

6. **downtime_dictionary**: with columns:
     - `downtime_reason_id` (int, AI, PK)
     - `downtime_reason` (varchar(50))

7.**factory_data**: with columns:
     - `factory_id` (int, AI, PK)
     - `factory_name` (varchar(255))
     - `factory_location` (varchar(255))
     - `office_id` (int)

8.**hmi_minder_connection_logs**: with columns:
     - `log_id` (int, AI, PK)
     - `connection_timestamp` (datetime)
     - `ip_address` (varchar(50))
     - `hmi_minder_connection_status` (varchar(10))

9.**edge_minder_connection_logs**: with columns:
   - `log_id` (int, AI, PK)
   - `minder_ip_address` (varchar(50))
   - `ip_address` (varchar(50))
   - `edge_minder_connection_status` (tinyint(1))
   - `timestamp` (timestamp)

10.**machine_process_data**: with columns:
   - `record_id` (int, AI, PK)
   - `ip_address` (varchar(50))
   - `timestamp` (datetime)
   - `shot_count` (float)
   - `cycle_time_seconds` (float)
   - `injection_time_seconds` (float)
   - `dosing_time_seconds` (float)
   - `dosing_stop_time_seconds` (float)
   - `melt_cushion_height` (float)
   - `switch_over_position` (float)
   - `mold_close_time_seconds` (float)
   - `mold_open_time_seconds` (float)
   - `zone_3_temperature_celsius` (float)
   - `nozzle_1_temperature_celsius` (float)
   - `feed_temperature_celsius` (float)
   - `zone_1_temperature_celsius` (float)
   - `zone_2_temperature_celsius` (float)
   - `zone_4_temperature_celsius` (float)
   - `oil_temperature_celsius` (float)
   - `melt_temperature_celsius` (float)
   - `nozzle_2_temperature_celsius` (float)
   - `switch_over_pressure_bar` (float)
   - `tonnage_kN` (float)
   - `mold_open_stop_seconds` (float)
   - `tonnage_build_time_seconds` (float)
   - `tonnage_release_time_seconds` (float)
   - `ejector_forward_time_seconds` (float)
   - `ejector_back_time_seconds` (float)
   - `minimum_melt_cushion_height` (float)
   - `injection_start_position` (float)
   - `peak_injection_pressure_bar` (float)
   - `mold_zone_1_temperature_celsius` (float)
   - `mold_zone_2_temperature_celsius` (float)
   - `mtc_temperature_celsius` (float)

11.**reject_part_analysis**: with columns:
- `record_id` (int, AI, PK)
- `ip_address` (varchar(255))
- `from_time` (datetime)
- `to_time` (datetime)
- `rejected_parts_count` (int)
- `rejection_approval_status` (int)
- `rejection_id` (int)
- `rejection_remarks` (varchar(255))

12.**rejection_dictionary**: with columns:
- `rejection_id` (int, AI, PK)
- `rejection_reason` (varchar(50))

13.**alarm_event_history**: with columns:  
   - `alarm_event_id` (int, AI, PK)  
   - `ip_address` (varchar(45))  
   - `alarm_set_timestamp` (datetime)  
   - `alarm_reset_timestamp` (datetime)  
   - `alarm_number` (int)  

14.**machine_performance_metrics_hourly**: with columns:
- `record_id` (int, AI, PK)
- `ip_address` (varchar(50))
- `from_time` (datetime)
- `to_time` (datetime)
- `productivity_percentage` (float)
- `quality_percentage` (float)
- `good_parts_count` (int)
- `reject_parts_count` (int)
- `reject_percentage` (float)
- `downtime_percentage` (float)
- `downtime_seconds` (int)
- `uptime_seconds` (int)
- `utilization_percentage` (float)
- `energy_consumption` (int)
- `overall_equipment_effectiveness_percentage` (float)
- `actual_shot_production` (int)
- `actual_part_production` (int)
- `planned_shot_production` (int)
- `planned_part_production` (int)


table explanation:

since ip_address is used as foreign key in most of the tables, whenever machine_name is referred kindly get the ip_address of the machine from machine_details table and use it across.

machine_details:
purpose of the table: This table contains the machine details for each and every machine.
below are table coloumn details:
machine_id -this is the machine index.
connection_timestamp -this the time at which the record is created.
machine_name -this is the machine name.
ip_address -this is the machine's ip address.
office_id -this is the office id of the machine.
factory_id -this is the factory id of the machine.
shop_floor_id -this is the shopfloor id at which the machine is present.
production_line_id -this is the line at which the machine is present in the shop_floor_id, totally there are 4 lines. Also known as line number.

alarm_frequency_log:
purpose of the table:This table contains the alarm numbers and the corresponding ip addresses along with the alarm count for each and every machine.
below are table coloumn details:
alarm_frequency_id -is the index number.
alarm_number -this is the alarm number of the machine, which is the foreign key used for alarm_dictionary table, alarm_frequency_log table and alarm_event_history table.
alarm_frequency -this is alarm count of the machine or the number of times the alarm has occurred.
ip_address -this is the machine's ip address.
alarm_set_timestamp -this is the time at which the alarm has occured.
Note: Never use Count() function to get the alarm count, always use alarm_frequency coloumn to get the alarm count.
Example:how many times alarm no 121 occured for im08 during aug ?
SQL:SELECT ac.alarm_frequency AS total_alarm_count FROM alarm_frequency_log ac JOIN machine_details m ON ac.ip_address = m.ip_address WHERE m.machine_name = 'im08'AND ac.alarm_number = 121 AND ac.alarm_set_timestamp BETWEEN '2024-08-01 00:00:00' AND '2024-08-31 23:59:59';
Example:how many times alarm no 1 has occurred for all machines?
SQL:SELECT SUM(ac.alarm_frequency) AS total_alarm_count FROM alarm_frequency_log ac JOIN machine_details m ON ac.ip_address = m.ip_address WHERE ac.alarm_number = 1;
Example:how many times Pipe cooling error occured for im08 during oct ?
SQL:SELECT ac.alarm_frequency AS total_alarm_count FROM alarm_frequency_log ac JOIN machine_details m ON ac.ip_address = m.ip_address JOIN alarm_dictionary ad ON ac.alarm_number = ad.alarm_number WHERE m.machine_name = 'IM08'AND ad.alarm_description = 'Pipe cooling error'AND ac.alarm_set_timestamp BETWEEN '2024--10-01 00:00:00' AND '2024-10-31 23:59:59';

alarm_dictionary:
purpose of the table: This table contains list of alarm numbers and corresponding description.Alarm message displays in the machine when a issue occures in a machine.
below are table coloumn details:
alarm_description_id- is the index number.
alarm_number- this is the alarm number of the machine, which is the foreign key used for alarm_dictionary table, alarm_frequency_log table and alarm_event_history table.
alarm_description- this is the alarm description. 
Example: what are the alarms that are present?
SQL:SELECT alarm_number, alarm_description FROM alarm_dictionary; 

alarm_event_history:
purpose of the table: This table contains the detials about the alarm number and corresponding ip address with alarm set time and reset time.
below are table coloumn details:
alarm_event_id -this is the index number.
ip_address -this is the machine's ip address.
alarm_set_timestamp -this is the time at which the alarm has occured.
alarm_reset_timestamp - this is the alarm reset time, has a default value of '2038-01-19 03:14:07'which means the alarm is not reset yet so ingore this.
alarm_number -this is the alarm number of the machine, which is the foreign key used for alarm_dictionary table, alarm_frequency_log table and alarm_event_history table.
Example: What are the alarms that are active? or What are the current alarms? or what are the alarm which are not reset?
SQL:SELECT md.machine_name,md.ip_address,aeh.alarm_number,ad.alarm_description,aeh.alarm_set_timestamp,CASE WHEN aeh.alarm_reset_timestamp = '2038-01-19 03:14:07' THEN 'Not Reset'ELSE aeh.alarm_reset_timestamp END AS alarm_reset_timestamp FROM machine_details md JOIN alarm_event_history aeh ON md.ip_address = aeh.ip_address JOIN alarm_dictionary ad ON aeh.alarm_number = ad.alarm_number WHERE aeh.alarm_reset_timestamp = '2038-01-19 03:14:07'; -- Filters for active alarms

downtime_logs:
purpose of the table:This table records downtime information for each machine.
below are table coloumn details:
downtime_id - The index of the record.
ip_address - this is the machine's ip address.
from_time - The production start time for the machine.
to_time - The production end time for the machine.
downtime_interval - The downtime duration in the format H:M:S(Hours:Minutes:seconds) within the from_time and to_time window.
downtime_reason_id - The reason id for the downtime, which is the foreign key used in downtime_logs and downtime_dictionary.
downtime_remarks - Additional remarks from the machine operator for the downtime.
downtime_approval - The approval status of the downtime. 1 means waiting for supervisor approval, 2 means waiting for the analytical engine to process, and 3 means the analytical engine has processed and updated the OEE.
downtime_category - This is the downtime reason category which is by default unplanned. And this can be changed to planned if the downtime is planned by the manager or the staff.

When downtime is requested, use the downtime_interval coloumn value only. do not use time difference of from_time and to_time if downtime is requested.

Example:
Question:what is the downtime of im06 during oct 2024?
SQL:SELECT machine_details.machine_name, SEC_TO_TIME(SUM(TIME_TO_SEC(downtime_logs.downtime_interval))) AS total_downtime FROM machine_details JOIN downtime_logs ON machine_details.ip_address = downtime_logs.ip_address WHERE machine_details.machine_name = 'IM06' AND downtime_logs.from_time >= '2024-10-01 00:00:00' AND downtime_logs.to_time <= '2024-10-31 23:59:59';
Example:
Question:which machine has got less downtime on 26th Oct 2024?
SQL:SELECT machine_details.machine_name, SUM(downtime_logs.downtime_interval) AS total_downtime FROM machine_details JOIN downtime_logs ON machine_details.ip_address = downtime_logs.ip_address WHERE downtime_logs.from_Time >= '2024-10-26 00:00:00' AND downtime_logs.to_Time <= '2024-10-26 23:59:59' GROUP BY machine_details.machine_name ORDER BY total_downtime ASC LIMIT 1;

downtime_dictionary:
purpose of the table: This table contains all the downtime reasons .
below are table coloumn details:
downtime_reason_id -The reason id for the downtime, which is the foreign key used in downtime_logs and downtime_dictionary.
downtime_reason - this is the down time reason.

factory_data:
purpose of the table: This table contains the factory details.
below are table coloumn details:
factory_id- this is the factory id.
factory_name -this is the factory name.
factory_location -this is the factory location.
office_id -this is the office id.

machine_parameters:
purpose of the table: This table contains the machine parameters of each and every machine.This data is updated periodically and are actual.
below are table coloumn details:
parameter_id -this is the index of the table.
timestamp -this the time at which the record is created.
ip_address -this is the machine's ip address.
machine_name -this is the machine name.
machine_number -this is the machine number.
machine_configuration -this is the machine configuration.
mold_id -this is the mold id that is used in the injection molding machine. Also known as mold, dye, cast, tool that is used in the injection molding machine.
part_number -this is the part number of the machine.
material_name -this is the material name, the material is used for the product.
machine_mode -this is the machine mode or status of machine. 4 - fully automatic. 3 - semi automatic. 2 - manual.
cavity_count -this is the cavity count of tha mold.
pump_status -this is the pump status of the machine.
heater_status -this is the heater status of the machine.
shot_weight -this is the shot weight.
actual_good_parts -this is the count of actual good parts produced by the machine which is incremented periodically.
actual_rejected_parts -this is the actual reject parts produced by the machine which is incremented periodically.
actual_good_shots -this is the actual good shot produced by the machine which is incremented periodically.
actual_total_shots -this is the actual total shot produced by the machine, which is the addition of actual_good_parts and actual_rejected_parts.
good_parts_target -this is the good part target to be produced by the machine.
energy_consumption -this energy consumption value by the machine.
ideal_cycle_time_seconds -this is the ideal cycle time for this machine for this mold.

example:Give the machines with pump on?
SQL:WITH machine_parameters_logs AS (SELECT ip_address, pump_status, timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY timestamp DESC) AS rn FROM machine_parameters)SELECT DISTINCT md.machine_name,CASE WHEN TIMESTAMPDIFF(SECOND, mpl.timestamp, NOW()) <= 300 AND mpl.pump_status = 1 THEN 'Running'WHEN TIMESTAMPDIFF(SECOND, mpl.timestamp, NOW()) <= 300 AND mpl.pump_status = 0 THEN 'Not Running'ELSE 'Current status Not Available'END AS pump_status FROM machine_details md LEFT JOIN machine_parameters_logs mpl ON md.ip_address = mpl.ip_address AND mpl.rn = 1;
example:Give the machines with heater on?
SQL:WITH machine_parameters_logs AS (SELECT ip_address, heater_status, timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY timestamp DESC) AS rn FROM machine_parameters)SELECT DISTINCT md.machine_name,CASE WHEN TIMESTAMPDIFF(SECOND, mpl.timestamp, NOW()) <= 300 AND mpl.heater_status = 1 THEN 'Running'WHEN TIMESTAMPDIFF(SECOND, mpl.timestamp, NOW()) <= 300 AND mpl.heater_status = 0 THEN 'Not Running'ELSE 'Current status Not Available'END AS heater_status FROM machine_details md LEFT JOIN machine_parameters_logs mpl ON md.ip_address = mpl.ip_address AND mpl.rn = 1;

When asked to calculate values for columns actual_good_parts,actual_rejected_parts,actual_good_shots,actual_total_shots (only these columns) always compute the difference as MAX(column) - MIN(column) within the specified time range instead of summing the values[never ever sum up these columns when asked for total].for these specific columns alone follow the example:

how many good parts, rejected parts, good shots, total shots, total parts produced for the partnumbers 4683, 75048, 50000488 for im08 in oct 2024?
SQL:WITH part_data AS (SELECT mp.part_number,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,mp.actual_good_shots,mp.actual_total_shots,LAG(mp.actual_good_parts) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_rejected_parts,LAG(mp.actual_good_shots) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_good_shots,LAG(mp.actual_total_shots) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_total_shots FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name = 'IM08'AND mp.timestamp BETWEEN '2024-10-01 00:00:00' AND '2024-10-31 23:59:59'AND mp.part_number IN ('4683', '75048', '50000488'))SELECT SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS total_good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS total_rejected_parts,SUM(CASE WHEN actual_good_shots >= prev_good_shots THEN actual_good_shots - prev_good_shots ELSE actual_good_shots END) AS total_good_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) AS total_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN (actual_total_shots - prev_total_shots) * cavity_count ELSE actual_total_shots * cavity_count END) AS total_parts FROM part_data WHERE prev_good_parts IS NOT NULL;
how many good parts and reject parts are produced for the each partnumbers 33608,50086420002RE,50000949003,7073 for im08 in dec 2024?
SQL:WITH part_data AS (SELECT mp.part_number,mp.timestamp,mp.actual_good_parts,mp.actual_rejected_parts,LAG(mp.actual_good_parts) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_rejected_parts FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name = 'IM08'AND mp.timestamp BETWEEN '2024-12-01 00:00:00' AND '2024-12-31 23:59:59'AND mp.part_number IN ('33608', '50086420002RE', '50000949003', '7073'))SELECT part_number,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS total_good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS total_rejected_parts FROM part_data WHERE prev_good_parts IS NOT NULL GROUP BY part_number;
give me the good parts of im08 during dec 2024?
SQL:WITH part_data AS (SELECT mp.part_number,mp.timestamp,mp.cavity_count,mp.actual_good_parts,LAG(mp.actual_good_parts) OVER (PARTITION BY mp.part_number ORDER BY mp.timestamp) AS prev_good_parts FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name = 'IM08'AND mp.timestamp BETWEEN '2024-12-01 00:00:00' AND '2024-12-31 23:59:59')SELECT SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS total_good_parts FROM part_data WHERE prev_good_parts IS NOT NULL;  
how many good parts,rejected parts, good shots, total shots, total parts produced for im08 with respect to mold during oct 2024? or give me the good parts,rejected parts, good shots, total shots, total parts for im08 in oct 2024?
SQL:WITH mold_data AS (SELECT md.machine_name,mp.mold_id,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,mp.actual_good_shots,mp.actual_total_shots,LAG(mp.actual_good_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_rejected_parts,LAG(mp.actual_good_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_shots,LAG(mp.actual_total_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_total_shots FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name = 'IM08' AND mp.timestamp BETWEEN '2024-12-01 00:00:00' AND '2024-12-31 23:59:59')SELECT machine_name,mold_id,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS rejected_parts,SUM(CASE WHEN actual_good_shots >= prev_good_shots THEN actual_good_shots - prev_good_shots ELSE actual_good_shots END) AS good_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) AS total_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) * cavity_count AS total_parts FROM mold_data WHERE prev_good_parts IS NOT NULL GROUP BY machine_name, mold_id, cavity_count;
how many good parts,reject parts are produced for im08, im09 for oct 2024?
SQL:WITH part_data AS (SELECT md.machine_name,mp.part_number,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,LAG(mp.actual_good_parts) OVER (PARTITION BY md.machine_name, mp.part_number ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY md.machine_name, mp.part_number ORDER BY mp.timestamp) AS prev_rejected_parts FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name IN ('IM08', 'IM09')AND mp.timestamp BETWEEN '2024-10-01 00:00:00' AND '2024-10-31 23:59:59')SELECT machine_name,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS total_good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS total_rejected_parts FROM part_data WHERE prev_good_parts IS NOT NULL GROUP BY machine_name;
how many good parts, reject parts are produced for each machine today?
SQL:WITH part_data AS (SELECT md.machine_name,mp.part_number,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,LAG(mp.actual_good_parts) OVER (PARTITION BY md.machine_name, mp.part_number ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY md.machine_name, mp.part_number ORDER BY mp.timestamp) AS prev_rejected_parts FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE DATE(mp.timestamp) = CURDATE())SELECT machine_name,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS total_good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS total_rejected_parts FROM part_data WHERE prev_good_parts IS NOT NULL GROUP BY machine_name;
what are the good parts, rejected parts, total parts for the molds KNOB,BATTERYTHROUG, RESIDEOBASE, COVERBOTTOM, YELLOWCOVER, WHITECOVER, WHITECV, cellholdertop, CELLHOLDERMID during oct 2024?
SQL:WITH mold_data AS (SELECT mp.mold_id,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,mp.actual_total_shots,LAG(mp.actual_good_parts) OVER (PARTITION BY mp.mold_id ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY mp.mold_id ORDER BY mp.timestamp) AS prev_rejected_parts,LAG(mp.actual_total_shots) OVER (PARTITION BY mp.mold_id ORDER BY mp.timestamp) AS prev_total_shots FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE mp.mold_id IN ('KNOB', 'BATTERYTHROUG', 'RESIDEOBASE', 'COVERBOTTOM', 'YELLOWCOVER', 'WHITECOVER', 'WHITECV', 'cellholdertop', 'CELLHOLDERMID')AND mp.timestamp BETWEEN '2024-10-01 00:00:00' AND '2024-10-31 23:59:59')SELECT mold_id,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS rejected_parts,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) * cavity_count AS total_parts FROM mold_data WHERE prev_good_parts IS NOT NULL GROUP BY mold_id, cavity_count;
give me the parts produced for im06 in oct 2024? or what is the total parts produced for im06 in octobar 2024?
SQL:WITH mold_data AS (SELECT md.machine_name,mp.mold_id,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,mp.actual_good_shots,mp.actual_total_shots,LAG(mp.actual_good_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_rejected_parts,LAG(mp.actual_good_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_shots,LAG(mp.actual_total_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_total_shots FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name = 'IM06' AND mp.timestamp BETWEEN '2024-10-01 00:00:00' AND '2024-10-31 23:59:59')SELECT SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN (actual_total_shots - prev_total_shots) * cavity_count ELSE actual_total_shots * cavity_count END) AS total_parts FROM mold_data WHERE prev_good_parts IS NOT NULL;
can you tell about the production details of im06 in october 2024?
SQL:WITH mold_data AS (SELECT md.machine_name,mp.mold_id,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,mp.actual_good_shots,mp.actual_total_shots,LAG(mp.actual_good_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_rejected_parts,LAG(mp.actual_good_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_shots,LAG(mp.actual_total_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_total_shots FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE md.machine_name = 'IM06' AND mp.timestamp BETWEEN '2024-12-01 00:00:00' AND '2024-12-31 23:59:59')SELECT machine_name,mold_id,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS rejected_parts,SUM(CASE WHEN actual_good_shots >= prev_good_shots THEN actual_good_shots - prev_good_shots ELSE actual_good_shots END) AS good_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) AS total_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) * cavity_count AS total_parts FROM mold_data WHERE prev_good_parts IS NOT NULL GROUP BY machine_name, mold_id, cavity_count;
give me the production details of all machines during dec? or what are the good parts,rejected parts, good shots, total shots, total parts for all machines during dec?
SQL:WITH mold_data AS (SELECT md.machine_name,mp.mold_id,mp.timestamp,mp.cavity_count,mp.actual_good_parts,mp.actual_rejected_parts,mp.actual_good_shots,mp.actual_total_shots,LAG(mp.actual_good_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_parts,LAG(mp.actual_rejected_parts) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_rejected_parts,LAG(mp.actual_good_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_good_shots,LAG(mp.actual_total_shots) OVER (PARTITION BY md.machine_name, mp.mold_id ORDER BY mp.timestamp) AS prev_total_shots FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE DATE_FORMAT(mp.timestamp, '%Y-%m') = '2024-12')SELECT machine_name,mold_id,SUM(CASE WHEN actual_good_parts >= prev_good_parts THEN actual_good_parts - prev_good_parts ELSE actual_good_parts END) AS good_parts,SUM(CASE WHEN actual_rejected_parts >= prev_rejected_parts THEN actual_rejected_parts - prev_rejected_parts ELSE actual_rejected_parts END) AS rejected_parts,SUM(CASE WHEN actual_good_shots >= prev_good_shots THEN actual_good_shots - prev_good_shots ELSE actual_good_shots END) AS good_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) AS total_shots,SUM(CASE WHEN actual_total_shots >= prev_total_shots THEN actual_total_shots - prev_total_shots ELSE actual_total_shots END) * cavity_count AS total_parts FROM mold_data WHERE prev_good_parts IS NOT NULL GROUP BY machine_name, mold_id, cavity_count;


hmi_minder_connection_logs:
purpose of the table: This table contains the machine's connectivity status that is whether the machine is on or off.
below are table coloumn details:
log_id -this is the index of the table.
connection_timestamp -this the time at which the status is updated.
ip_address -this is the machine's ip address.
hmi_minder_connection_status -this is the ftp status of the machine, its either 0 or 1. If 0 it means the machine is off, if 1 it means the machine is on. Also knows as hmi minder connection status.

edge_minder_connection_logs:
purpose of the table: This table contains the data about minder connectivity that is wifi status of each and every machine.
below are table coloumn details:
log_id -this is the index of the table.
minder_ip_address -this is the ip address of the minder or wifi. 
ip_address -this is the machine's ip address.
edge_minder_connection_status -this is the status of the minder or wifi. Which is either 0 or 1. 0 means it is disconnected and 1 means it is connected.
timestamp -this the time at which the status is updated.

machine_process_data:
purpose of this table: this table conatins machine's parameter details. Note that this table is different from machine_parameters table. This table is updated periodically and are actual. 
below are table coloumn details:
record_id -this is the index of the table.
ip_address -this is the machine's ip address.
timestamp- the time at which the data is added to the table.
shot_count -this is the shot count produced by the machine.
cycle_time_seconds -this is the cycle time of the machine in seconds.
injection_time_seconds -this is the injection time of the machine in seconds.
dosing_time_seconds -this is the dosing time of the machine in seconds.
dosing_stop_time_seconds -this is the dosing stop time of the machine in seconds.
melt_cushion_height -this is the melt cushion height of the machine.
switch_over_position -this is the switch over position of the machine.
mold_close_time_seconds -this is the time at which the machine's mold open in seconds.
mold_open_time_seconds -this is the time at which the machine's mold close in seconds.
zone_3_temperature_celsius - this is the zone 3 temperature of the machine in celsius.
nozzle_1_temperature_celsius -this is the nozzle 1 temperature of the machine in celsius. 
feed_temperature_celsius -this is the feed temperature of the machine in celsius.
zone_1_temperature_celsius -this is the zone 1 temperature of the machine in celsius.
zone_2_temperature_celsius -this is the zone 2 temperature of the machine in celsius.
zone_4_temperature_celsius -this is the zone 4 temperature of the machine in celsius.
oil_temperature_celsius - this is the oil temperature of the machine in celsius.
melt_temperature_celsius -this is the melt temperature of the machine in Celsius.
nozzle_2_temperature_celsius -this is the nozzle 2 temperature of the machine in celsius.
switch_over_pressure_bar -this is the switch over presure of the machine in bar.
tonnage_kN -this is the tonnage of the machine in kilonewtons (kN).
mold_open_stop_seconds -this is the mold open stop time of the machine in seconds.
tonnage_build_time_seconds -this is the tonnage build time of the machine in seconds.
tonnage_release_time_seconds -this is the tonnage release time of the machine in seconds.
ejector_forward_time_seconds -this is the ejector forward time of the machine in seconds.
ejector_back_time_seconds -this is the ejector back time of the machine in seconds.
minimum_melt_cushion_height -this is the minimum melt cushion height of the machine.
injection_start_position -this is the injection start position of the machine.
peak_injection_pressure_bar -this is the peak injection pressure of the machine in bar.
mold_zone_1_temperature_celsius -this is the mold zone 1 temperature of the machine in celsius.
mold_zone_2_temperature_celsius -this is the mold zone 2 temperature of the machine in celsius.
mtc_temperature_celsius -this is the mtc(Mold Temperature Controller) temperature of the machine in celsius.
Example:Give me the record ID, IP address, timestamp, shot count, cycle time, injection time, dosing time, dosing stop time, melt cushion height, switch over position, mold close time, mold open time, zone 3 temperature, nozzle 1 temperature, feed temperature, zone 1 temperature, zone 2 temperature, zone 4 temperature, oil temperature, melt temperature, nozzle 2 temperature, switch over pressure, tonnage, mold open stop time, tonnage build time, tonnage release time, ejector forward time, ejector back time, minimum melt cushion height, injection start position, peak injection pressure, mold zone 1 temperature, mold zone 2 temperature, and MTC temperature for im06 during October 2024?
SQL:SELECT mpd.record_id,mpd.ip_address,mpd.timestamp,mpd.shot_count,mpd.cycle_time_seconds,mpd.injection_time_seconds,mpd.dosing_time_seconds,mpd.dosing_stop_time_seconds,mpd.melt_cushion_height,mpd.switch_over_position,mpd.mold_close_time_seconds,mpd.mold_open_time_seconds,mpd.zone_3_temperature_celsius,mpd.nozzle_1_temperature_celsius,mpd.feed_temperature_celsius,mpd.zone_1_temperature_celsius,mpd.zone_2_temperature_celsius,mpd.zone_4_temperature_celsius,mpd.oil_temperature_celsius,mpd.melt_temperature_celsius,mpd.nozzle_2_temperature_celsius,mpd.switch_over_pressure_bar,mpd.tonnage_kN,mpd.mold_open_stop_seconds,mpd.tonnage_build_time_seconds,mpd.tonnage_release_time_seconds,mpd.ejector_forward_time_seconds,mpd.ejector_back_time_seconds,mpd.minimum_melt_cushion_height,mpd.injection_start_position,mpd.peak_injection_pressure_bar,mpd.mold_zone_1_temperature_celsius,mpd.mold_zone_2_temperature_celsius,mpd.mtc_temperature_celsius FROM machine_process_data mpd JOIN machine_details md ON mpd.ip_address = md.ip_address WHERE md.machine_name = 'IM06' AND mpd.timestamp BETWEEN '2024-10-01' AND '2024-10-31';
Example: what is the tonnage for im08 now?
SQL:SELECT CASE WHEN TIMESTAMPDIFF(SECOND, mpd.timestamp, NOW()) <= 300 THEN mpd.tonnage_kN ELSE 'Current status not available' END AS tonnage_kN FROM machine_process_data mpd JOIN machine_details md ON mpd.ip_address = md.ip_address WHERE md.machine_name = 'IM08' ORDER BY mpd.timestamp DESC LIMIT 1;

reject_part_analysis:
purpose of the table: This table contains details about rejects, reject reasons, reject remarks and approval for each machine.This table is updated hourly.
below are table coloumn details:
record_id - index of the table.
ip_address -this is the machine's ip address.
from_time -this is the start of time window.
to_time -this is the end of time window.
rejected_parts_count -this is the number of reject parts of the machine.
rejection_approval_status -this is the number of approval of the machine.
rejection_id -this is the rejection id, which is the foreign key for reject_part_analysis table and rejection_dictionary table.
rejection_remarks -this is the remarks for the rejection.

rejection_dictionary:
purpose of the table: This table contains all the rejection reasons.
rejection_id - this is the rejection id, which is the foreign key for reject_part_analysis table and rejection_dictionary table.
rejection_reason -Reason for the rejection.

machine_performance_metrics_hourly:
purpose of the table: This table contains details about hourly performance analytics of all the machine.The table is populated one hour once for each machine.
record_id -this is the index of the table.
ip_address -this is the machine's ip address.
from_time -this is the start of time window.
to_time -this is the end of time window.
productivity_percentage -this is the productivity of the machine. Also known as availability of the machine.
quality_percentage -this is the quality of the machine.
good_parts_count -this is the count of good parts produced by the machine.
reject_parts_count -this is the count of rejected parts produced by the machine.
reject_percentage -this is the percentage of parts that were rejected.
downtime_percentage -this is the percentage of time the machine was down.
downtime_seconds -this is the total downtime in seconds.
uptime_seconds -this is the total uptime in seconds.
utilization_percentage -this is the utilization percentage of the machine.
energy_consumption -this is the energy consumption of the machine.
overall_equipment_effectiveness_percentage - This is the overall equipment effectiveness of the machine. Also known as OEE of the machine.
actual_shot_production -this is the actual number of shots produced by the machine.
actual_part_production -this is the actual number of parts produced by the machine.
planned_shot_production -this is the planned number of shots to be produced by the machine.
planned_part_production -this is the planned number of parts to be produced by the machine.

While using the table machine_performance_metrics_hourly always use `ORDER BY from_time`.
When answering questions related to analytics or columns present in machine_performance_metrics_hourly use:
Example: What is the OEE of IM06 on 25th Oct 2024 from 8am to 10am?
SQL:SELECT machine_details.machine_name,machine_performance_metrics_hourly.from_time,machine_performance_metrics_hourly.to_time, machine_performance_metrics_hourly.overall_equipment_effectiveness_percentage FROM machine_details JOIN machine_performance_metrics_hourly ON machine_details.ip_address = machine_performance_metrics_hourly.ip_address WHERE machine_performance_metrics_hourly.from_time >= '2024-10-25 08:00:00' AND machine_performance_metrics_hourly.to_time <= '2024-10-25 10:00:00' AND machine_details.machine_name = 'IM06' ORDER BY machine_performance_metrics_hourly.from_time;
Example: Can i get the hourly analytics of im08 during oct 2024?
SQL:SELECT * FROM machine_performance_metrics_hourly WHERE ip_address = (SELECT ip_address FROM machine_details WHERE machine_name = 'im08') AND from_time >= '2024-10-01' AND to_time <= '2024-10-31 23:59:59'ORDER BY from_time;
Example :show me the trend in oee of all machines in this month
SQL:SELECT md.machine_name, mpmh.from_time, mpmh.to_time, mpmh.overall_equipment_effectiveness_percentage FROM machine_details AS md JOIN machine_performance_metrics_hourly AS mpmh ON md.ip_address = mpmh.ip_address WHERE mpmh.from_time >= DATE_FORMAT(CURDATE(), '%Y-%m-01') AND mpmh.to_time <= LAST_DAY(CURDATE()) ORDER BY md.machine_name, mpmh.from_time;

while finding average of overall_equipment_effectiveness_percentage, productivity_percentage,quality_percentage, utilization_percentage columns present in machine_performance_metrics_hourly use the below example (strictly follow the below example):
Example : give me the avg oee, productivity, utilization and quality of im08 from march 13 to march 15 2025?
SQL:WITH Metrics AS (SELECT COUNT(*) AS TotalRecords,ROUND(AVG(CASE WHEN mpm.productivity_percentage > 0 THEN mpm.productivity_percentage ELSE NULL END), 2) AS avg_productivity_percentage,ROUND((SUM(mpm.good_parts_count) * 100.0 / NULLIF(SUM(mpm.actual_part_production), 0)), 2) AS avg_quality_percentage,ROUND(AVG(CASE WHEN mpm.utilization_percentage > 0 THEN mpm.utilization_percentage WHEN mpm.utilization_percentage = 0 AND EXISTS (SELECT 1 FROM downtime_logs dt WHERE dt.from_time = mpm.from_time AND dt.to_time = mpm.to_time AND dt.downtime_category = 'unplanned'AND NOT EXISTS (SELECT 1 FROM downtime_logs dt2 WHERE dt2.from_time = mpm.from_time AND dt2.to_time = mpm.to_time AND dt2.downtime_category != 'unplanned'AND dt2.downtime_approval = 3)) THEN mpm.utilization_percentage ELSE NULL END), 2) AS avg_utilization_percentage FROM machine_performance_metrics_hourly AS mpm JOIN machine_details AS md ON md.ip_address = mpm.ip_address WHERE md.machine_name = 'im08' AND mpm.from_time >= '2025-03-13 00:00:00' AND mpm.to_time <= '2025-03-15 00:00:00')SELECT avg_productivity_percentage,avg_quality_percentage,avg_utilization_percentage,ROUND((avg_utilization_percentage * avg_quality_percentage * avg_productivity_percentage / 10000), 2) AS avg_oee_percentage FROM Metrics;
Example :give me the avg oee, productivity, utilization and quality of all machines from march 13 to march 15 2025?
SQL:WITH Metrics AS (SELECT md.machine_name, COUNT(*) AS TotalRecords,ROUND(AVG(CASE WHEN mpm.productivity_percentage > 0 THEN mpm.productivity_percentage ELSE NULL END), 2) AS avg_productivity_percentage,ROUND((SUM(mpm.good_parts_count) * 100.0 / NULLIF(SUM(mpm.actual_part_production), 0)), 2) AS avg_quality_percentage,ROUND(AVG(CASE WHEN mpm.utilization_percentage > 0 THEN mpm.utilization_percentage WHEN mpm.utilization_percentage = 0 AND EXISTS ( SELECT 1 FROM downtime_logs dt WHERE dt.from_time = mpm.from_time AND dt.to_time = mpm.to_time AND dt.downtime_category = 'unplanned' AND NOT EXISTS ( SELECT 1 FROM downtime_logs dt2 WHERE dt2.from_time = mpm.from_time AND dt2.to_time = mpm.to_time AND dt2.downtime_category != 'unplanned'AND dt2.downtime_approval = 3)) THEN mpm.utilization_percentage ELSE NULL END), 2) AS avg_utilization_percentage FROM machine_performance_metrics_hourly AS mpm JOIN machine_details AS md ON md.ip_address = mpm.ip_address WHERE mpm.from_time >= '2025-03-13 00:00:00'AND mpm.to_time <= '2025-03-15 00:00:00' GROUP BY md.machine_name) SELECT machine_name, avg_productivity_percentage,avg_quality_percentage,avg_utilization_percentage,ROUND((avg_utilization_percentage * avg_quality_percentage * avg_productivity_percentage / 10000), 2) AS avg_oee_percentage FROM Metrics ORDER BY machine_name;
Example: give me the oee, productivity, quality and utilization for all machines in march 2025 day wise?
SQL:SELECT md.machine_name, DATE(mpm.from_time) AS Date, ROUND(AVG(CASE WHEN mpm.productivity_percentage > 0 THEN mpm.productivity_percentage ELSE NULL END), 2) AS avg_productivity_percentage, ROUND((SUM(mpm.good_parts_count) * 100.0 / NULLIF(SUM(mpm.actual_part_production), 0)), 2) AS avg_quality_percentage, ROUND(AVG(CASE WHEN mpm.utilization_percentage > 0 THEN mpm.utilization_percentage WHEN mpm.utilization_percentage = 0 AND EXISTS ( SELECT 1 FROM downtime_logs dt WHERE dt.from_time = mpm.from_time AND dt.to_time = mpm.to_time AND dt.downtime_category = 'unplanned' AND NOT EXISTS ( SELECT 1 FROM downtime_logs dt2 WHERE dt2.from_time = mpm.from_time AND dt2.to_time = mpm.to_time AND dt2.downtime_category != 'unplanned' AND dt2.downtime_approval = 3)) THEN mpm.utilization_percentage ELSE NULL END), 2) AS avg_utilization_percentage, ROUND((AVG(CASE WHEN mpm.utilization_percentage > 0 THEN mpm.utilization_percentage WHEN mpm.utilization_percentage = 0 AND EXISTS ( SELECT 1 FROM downtime_logs dt WHERE dt.from_time = mpm.from_time AND dt.to_time = mpm.to_time AND dt.downtime_category = 'unplanned' AND NOT EXISTS ( SELECT 1 FROM downtime_logs dt2 WHERE dt2.from_time = mpm.from_time AND dt2.to_time = mpm.to_time AND dt2.downtime_category != 'unplanned' AND dt2.downtime_approval = 3)) THEN mpm.utilization_percentage ELSE NULL END) * (SUM(mpm.good_parts_count) * 100.0 / NULLIF(SUM(mpm.actual_part_production), 0)) * AVG(CASE WHEN mpm.productivity_percentage > 0 THEN mpm.productivity_percentage ELSE NULL END) / 10000), 2) AS avg_oee_percentage FROM machine_performance_metrics_hourly AS mpm JOIN machine_details AS md ON md.ip_address = mpm.ip_address WHERE mpm.from_time >= '2025-03-01 00:00:00' AND mpm.to_time <= '2025-03-31 23:59:59' GROUP BY md.machine_name, DATE(mpm.from_time) ORDER BY md.machine_name, DATE(mpm.from_time);

Edge case questions:
1.What are the machines that are running? or What are the machines that are not running?
SQL:WITH latest_edge_minder AS (SELECT ip_address, minder_ip_address, edge_minder_connection_status,timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY timestamp DESC) AS rn FROM edge_minder_connection_logs),latest_hmi_minder AS (SELECT ip_address, hmi_minder_connection_status,connection_timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY connection_timestamp DESC) AS rn FROM hmi_minder_connection_logs),recent_machine_parameters AS (SELECT ip_address,machine_mode,pump_status,heater_status,timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY timestamp DESC) AS rn FROM machine_parameters WHERE timestamp >= NOW() - INTERVAL 300 SECOND)SELECT md.machine_name,md.ip_address,CASE WHEN em.edge_minder_connection_status = 0 OR hm.hmi_minder_connection_status = 0 THEN 'grey' WHEN em.edge_minder_connection_status = 1 AND hm.hmi_minder_connection_status = 1 THEN CASE WHEN mp.pump_status = 0 OR mp.heater_status = 0 THEN 'red' WHEN mp.machine_mode = 3 THEN 'yellow' WHEN mp.machine_mode = 4 THEN 'green' ELSE 'grey' END ELSE 'grey'END AS color_code, CASE WHEN em.edge_minder_connection_status = 0 THEN 'Current status not available' WHEN hm.hmi_minder_connection_status = 0 THEN 'Current status not available' WHEN mp.ip_address IS NULL THEN 'Current status not available' WHEN mp.machine_mode IN (3, 4) AND mp.pump_status = 1 AND mp.heater_status = 1 THEN 'Machine running' ELSE 'Machine not running' END AS machine_status FROM machine_details md LEFT JOIN latest_edge_minder em ON md.ip_address = em.ip_address AND em.rn = 1 LEFT JOIN latest_hmi_minder hm ON md.ip_address = hm.ip_address AND hm.rn = 1 LEFT JOIN recent_machine_parameters mp ON md.ip_address = mp.ip_address AND mp.rn = 1;
2.Is im06 running?
SQL:WITH latest_edge_minder AS (SELECT ip_address, minder_ip_address, edge_minder_connection_status, timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY timestamp DESC) AS rn FROM edge_minder_connection_logs),latest_hmi_minder AS (SELECT ip_address, hmi_minder_connection_status, connection_timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY connection_timestamp DESC) AS rn FROM hmi_minder_connection_logs),recent_machine_parameters AS (SELECT ip_address,machine_mode,pump_status,heater_status,timestamp,ROW_NUMBER() OVER (PARTITION BY ip_address ORDER BY timestamp DESC) AS rn FROM machine_parameters WHERE timestamp >= NOW() - INTERVAL 300 SECOND)SELECT md.machine_name,md.ip_address,CASE WHEN em.edge_minder_connection_status = 0 OR hm.hmi_minder_connection_status = 0 THEN 'grey' WHEN em.edge_minder_connection_status = 1 AND hm.hmi_minder_connection_status = 1 THEN CASE WHEN mp.pump_status = 0 OR mp.heater_status = 0 THEN 'red' WHEN mp.machine_mode = 3 THEN 'yellow' WHEN mp.machine_mode = 4 THEN 'green' ELSE 'grey' END ELSE 'grey'END AS color_code, CASE WHEN em.edge_minder_connection_status = 0 THEN 'Current status not available' WHEN hm.hmi_minder_connection_status = 0 THEN 'Current status not available' WHEN mp.ip_address IS NULL THEN 'Current status not available' WHEN mp.machine_mode IN (3, 4) AND mp.pump_status = 1 AND mp.heater_status = 1 THEN 'Machine running' ELSE 'Machine not running' END AS machine_status FROM machine_details md LEFT JOIN latest_edge_minder em ON md.ip_address = em.ip_address AND em.rn = 1 LEFT JOIN latest_hmi_minder hm ON md.ip_address = hm.ip_address AND hm.rn = 1 LEFT JOIN recent_machine_parameters mp ON md.ip_address = mp.ip_address AND mp.rn = 1 WHERE md.machine_name = 'IM06';
3.which machine was running yesterday?
SQL:SELECT md.machine_name FROM machine_details md JOIN machine_parameters mp ON md.ip_address = mp.ip_address WHERE DATE(mp.timestamp) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND mp.machine_mode IN (3, 4) AND mp.pump_status = 1 AND mp.heater_status = 1 GROUP BY md.machine_name;
4.When was the last time im08 was running?
SQL:SELECT md.machine_name, MAX(mp.timestamp) AS last_run_time FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE mp.pump_status = 1 AND mp.heater_status = 1 AND mp.machine_mode IN (3, 4) AND md.machine_name = 'im08'GROUP BY md.machine_name;
5.When was the last time each machine ran?
SQL:SELECT md.machine_name, MAX(mp.timestamp) AS last_run_time FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE mp.pump_status = 1 AND mp.heater_status = 1 AND mp.machine_mode IN (3, 4) GROUP BY md.machine_name;
6.which machines ran today?
SQL:SELECT md.machine_name, MAX(mp.timestamp) AS last_run_time FROM machine_parameters mp JOIN machine_details md ON mp.ip_address = md.ip_address WHERE mp.pump_status = 1 AND mp.heater_status = 1 AND mp.machine_mode IN (3, 4) AND DATE(mp.timestamp) = CURDATE() GROUP BY md.machine_name;
7.give me avg, min, max for latest 10 cycle time of im06?
SQL:SELECT AVG(cycle_time_seconds) AS avg_cycle_time, MIN(cycle_time_seconds) AS min_cycle_time, MAX(cycle_time_seconds) AS max_cycle_time FROM (SELECT cycle_time_seconds FROM machine_process_data WHERE ip_address = (SELECT ip_address FROM machine_details WHERE machine_name = 'IM06') ORDER BY timestamp DESC LIMIT 10) AS latest_10_cycles;
8. what is the cycle time at which YELLOWCOVER mold ran in im08 in dec?
SQL:SELECT AVG(mpd.cycle_time_seconds) AS average_cycle_time FROM machine_process_data mpd JOIN machine_details md ON mpd.ip_address = md.ip_address JOIN machine_parameters mp ON mp.ip_address = mpd.ip_address AND mp.timestamp = mpd.timestamp WHERE md.machine_name = 'IM08' AND mp.mold_id = 'YELLOWCOVER' AND DATE_FORMAT(mpd.timestamp, '%Y-%m') = '2024-12';

if any question specifies with respect to month use DATE_FORMAT function dont use STRFTIME.
if the question doesn't specify any year use the current year which is '2024'.
Always join with machines table to get machine names when needed.
For current data(now),don't use timestamp DESC LIMIT 1, use the NOW()) <= 300 and if the data is present give the data else give 'Current status not available'.
"""

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for chat context
if 'context' not in st.session_state:
    st.session_state.context = {
        'last_query_type': None,
        'last_entities': {},
        'last_sql': None
    }

#The clean_sql_query(query) function sanitizes and formats an SQL query by removing unnecessary spaces, standardizing syntax, and preventing SQL injection risks.
def clean_sql_query(query):
    # Remove 'sql' prefix if it exists
    query = re.sub(r'^sql\s+', '', query.strip(), flags=re.IGNORECASE)
    
    # Remove backticks
    query = query.replace('`', '')
    
    # Clean up whitespace and formatting
    cleaned = ' '.join(query.split())
    cleaned = re.sub(r'(?i)(\b(SELECT|FROM|WHERE|AND|OR|JOIN|ON|GROUP BY|ORDER BY|LIMIT)\b)', r' \1 ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.replace('`', '')
    return cleaned

#The extract_query_context() function extracts key components from an SQL query, such as table names, selected columns, and conditions, to provide contextual information about the query.
def extract_query_context(sql_query):
    """Extract context from SQL query for future reference"""
    context = {
        'query_type': None,
        'entities': {},
        'columns': []
    }
    
    # Extract SELECT columns
    if 'SELECT' in sql_query.upper():
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)
        if select_match:
            context['columns'] = [col.strip() for col in select_match.group(1).split(',')]
    
    # Extract main table
    from_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
    if from_match:
        context['main_table'] = from_match.group(1)
    
    # Extract WHERE conditions
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+(?:GROUP|ORDER|LIMIT)|$)', sql_query, re.IGNORECASE)
    if where_match:
        context['conditions'] = where_match.group(1)
    
    return context

#The find_similar_question_advanced() function searches for similar questions in a JSON file based on a similarity threshold. It helps find relevant past queries by comparing their similarity to the given question.
def find_similar_question_advanced(json_file, question, threshold=0.6):
    """
    Searches through the JSON file for a similar question with advanced matching techniques.
    Uses multiple similarity metrics and optimizes for speed.
    
    :param json_file: Path to the JSON file containing multiple records.
    :param question: The input question to search for.
    :param threshold: Similarity threshold for matching (default 0.6).
    :return: Tuple (Matching Question, SQL Query) if a match is found, otherwise (None, None).
    """
    start_time = time.time()
    
    # Load the data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract valid question-SQL pairs
    questions = {entry["translated"]: entry["sql_query"] for entry in data 
                if "translated" in entry and "sql_query" in entry 
                and isinstance(entry["translated"], str) 
                and isinstance(entry["sql_query"], str)}
    
    if not questions:
        return None, None
    
    # Preprocess the input question
    processed_input = preprocess_text(question)
    
    # Initialize stemmer for word root comparison
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Get important terms from the input question
    input_terms = extract_important_terms(processed_input, stemmer, stop_words)
    
    # First pass: Fast filtering using keyword matching to reduce candidates
    candidates = {}
    for q, sql in questions.items():
        processed_q = preprocess_text(q)
        q_terms = extract_important_terms(processed_q, stemmer, stop_words)
        overlap_score = calculate_term_overlap(input_terms, q_terms)
        
        if overlap_score > threshold * 0.7:  # Lower threshold for first pass
            candidates[q] = {
                'sql': sql,
                'processed': processed_q,
                'terms': q_terms,
                'overlap_score': overlap_score
            }
    
    if not candidates:
        return None, None
    
    # Second pass: More detailed similarity for the remaining candidates
    best_match_question = None
    best_match_sql = None
    best_score = threshold
    
    for q, info in candidates.items():
        similarity = combined_similarity(
            processed_input, 
            info['processed'],
            input_terms,
            info['terms'],
            info['overlap_score']
        )
        
        if similarity > best_score:
            best_score = similarity
            best_match_question = q
            best_match_sql = info['sql']
    
    # If no valid match was found, return (None, None)
    if best_match_sql is None:
        return None, None
    
    return best_match_question, best_match_sql

#The preprocess_text() function cleans and normalizes text by removing special characters, converting to lowercase, and eliminating extra spaces. This helps in preparing text for further processing like NLP tasks or similarity matching.
def preprocess_text(text):
    """Preprocess text for better matching"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace special database/SQL terms with standardized versions
    replacements = {
        'ip address': 'ipaddress',
        'ip addr': 'ipaddress',
        'ip': 'ipaddress',
        'database': 'db',
        'count': 'number',
        # Add more domain-specific replacements as needed
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove punctuation except meaningful ones in DB queries
    text = re.sub(r'[^\w\s.=><]', '', text)
    
    return text

#The extract_important_terms() function extracts key terms from the given text by removing stop words and applying stemming. This helps in reducing text complexity while preserving essential meaning for NLP tasks like search or classification.
def extract_important_terms(text, stemmer, stop_words):
    """Extract and stem important terms from the text"""
    # Split into words
    words = re.findall(r'\b\w+\b', text)
    
    # Filter out stop words and stem the rest
    terms = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 1]
    
    # Create counter with frequency
    return Counter(terms)

#The calculate_term_overlap() function computes the overlap between two sets of terms, measuring how many terms are shared. This is useful for assessing text similarity in NLP tasks.
def calculate_term_overlap(terms1, terms2):
    """Calculate overlap score between two term sets"""
    # Find common terms
    common_terms = set(terms1.keys()) & set(terms2.keys())
    
    if not common_terms:
        return 0
    
    # Calculate score based on frequency of common terms
    overlap_score = sum(min(terms1[term], terms2[term]) for term in common_terms)
    total_terms = sum(terms1.values()) + sum(terms2.values())
    
    return (2 * overlap_score) / total_terms if total_terms > 0 else 0

#The combined_similarity() function calculates a similarity score by combining text-based similarity (e.g., string matching) and term overlap. This helps in measuring the overall similarity between two texts for NLP applications.
def combined_similarity(text1, text2, terms1, terms2, overlap_score):
    """Calculate combined similarity using multiple metrics"""
    # Sequential matching score (like difflib but optimized)
    seq_sim = SequenceMatcher(None, text1, text2).ratio()
    
    # Entity recognition (check for specific database entities)
    entity_match = entity_recognition_score(text1, text2)
    
    # Combine scores with weights
    combined = (
        0.5 * overlap_score +  # Term overlap is important
        0.3 * seq_sim +        # Sequential matching for phrase structure
        0.2 * entity_match     # Entity matching for database specific terms
    )
    
    return combined

#The entity_recognition_score() function evaluates the similarity between two texts based on named entity recognition (NER). It identifies and compares entities like names, locations, and organizations to measure how closely the texts are related.
def entity_recognition_score(text1, text2):
    """Identify and compare database-specific entities in both texts"""
    # Look for patterns like table names, column references, numbers
    patterns = [
        r'\b\w+\d+\b',              # Words with numbers (like server1, imp8)
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
        r'\b(select|from|where|join|group by|order by)\b',  # SQL keywords
        r'\b\w+\.\w+\b'             # Table.column patterns
    ]
    
    score = 0
    for pattern in patterns:
        entities1 = set(re.findall(pattern, text1))
        entities2 = set(re.findall(pattern, text2))
        
        if entities1 and entities2:
            # Calculate Jaccard similarity for entities
            intersection = len(entities1 & entities2)
            union = len(entities1 | entities2)
            score += intersection / union if union > 0 else 0
    
    # Normalize score
    return score / len(patterns) if patterns else 0

#The get_gemini_response() function interacts with the Gemini model to generate a SQL. It utilizes the current question, chat history, and context, along with optional inputs like previous SQL queries, errors, and similar questions to provide a more relevant and informed SQL.
def get_gemini_response(question, chat_history, current_context, previous_sql=None, previous_error=None,similar_question=None,similar_sql=None):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Construct context-aware prompt
        context_prompt = f"""Previous context: {current_context}
        Chat history: {chat_history[-10:] if chat_history else 'No previous context'}
        Current question: {question}
        
         Based on the previous context and chat history, generate an appropriate SQL query.
        IMPORTANT: 
        1. Return ONLY the raw SQL query without any prefixes, markdown, or additional text
        2. Do not include the word 'sql' or backticks in your response
        3. For questions about current data, make sure to use the NOW()) <= 300.
        4. Always join with the machines table when machine names are needed
        5. For analytics queries, use the appropriate analytics table based on the time frame
        """
        # If there's a previous error, refine the query
        if previous_sql and previous_error:
            context_prompt += f"""

            Previous SQL query:
            {previous_sql}
            
            The query failed with the following error:
            {previous_error}
            
            A similar question was found based on word similarity:
            {similar_question}, with the corresponding SQL: {similar_sql}
            
            However, this similar question might not be relevant to the current context.  
            Use {similar_question} and {similar_sql} **only if** they align with the same context as the previous SQL query.  
            
            Otherwise, ignore the similar question and SQL and focus on modifying the previous SQL to fix the error while preserving the intent of the original question.
            """
        generation_config = {
            'temperature': 0.1,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048
        }

        response = model.generate_content(
            [SYSTEM_PROMPT, context_prompt],
            generation_config=generation_config
        )
        response_text = response.text if hasattr(response, 'text') else ""
        
        # Remove any 'sql' prefix and clean up the query
        query = response_text.strip()
        if query.lower().startswith('sql'):
            query = query[3:].strip()
            
        return query
    
    except Exception as e:
        print("Error in generating response:", e)
        return "An error occurred. Please try again."

#The read_sql_query() function executes an SQL query on the given database connection. It handles query retries in case of failures and incorporates additional context from the question and translated speech to improve error handling or debugging.
def read_sql_query(sql, db,question,translated_speech,retry_count=0, max_retries=2):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=db
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [i[0] for i in cur.description]
        conn.close()
        return rows, columns,sql
    except Exception as e:
        error_code = e.errno
        error_type = type(e).__name__  # Get the error type
        print(f"[{error_type}] Database query error: {e}")

        if error_code == 3024:
            log_error(question, translated_speech, sql, error_type, e)
            return error_code, e, sql
        
        if retry_count < max_retries:
            json_file_path = "query_logs.json"
            matching_question,sql_result = find_similar_question_advanced(json_file_path, translated_speech)
            if sql_result:
                print(F"Matching Question: {matching_question}")
                print(f"Matching SQL Query: {sql_result}")
            else:
                print("No matching question found.")
            refined_sql = get_gemini_response(question, None, None, previous_sql=sql, previous_error=str(e),similar_question=matching_question,similar_sql=sql_result)
            
            if refined_sql:
                refined_sql = clean_sql_query(refined_sql)
                print(f"Refined SQL query (attempt {retry_count + 1}): {refined_sql}")
                return read_sql_query(refined_sql, db, question,translated_speech, retry_count=retry_count + 1, max_retries=max_retries)
            
        log_error(question,translated_speech,sql,error_type,e)
        return error_type, e,sql

#The generate_explanation() function generates a natural language explanation for the SQL query result. It uses the provided DataFrame, question, SQL query, and chat history to create a contextual response in the specified language.
def generate_explanation(df, question,sql,language,chat_history):
    """Generate a natural language explanation of the query results"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Convert DataFrame to string representation
    data_str = df.to_string()
    
    prompt = f"""
    You are a helpful AI assistant for a manufacturing database.
    Chat history: {chat_history[-10:] if chat_history else 'No previous context'}
    Current question: {question}
    Generated SQL :{sql}
    Based on the database query, here are the results:{data_str}
    use {language} language. 
    Answer the user's question directly based on the data. If the data doesn't fully answer it, provide what insights you can. After answering, explain the data with the context of the question in simple terms (1-2 sentences), focusing on the business meaning rather than technical terms in a paragraph.
    Dont provide tables as output, use paragraph.
    Note:
    1.Dont talk what the sql query does, it is just for understanding what data is shown to the user.Be more formal and professional.
    2.Dont give subheadings like answer or explanation give it in paragraph.
    3.if the question states to explain or analyse, explain or analyse about the out put in 100 to 200 words.
    4.Dont mention like this alarm_reset_timestamp of '2038-01-19 03:14:07', just say alarm is not reset.
    5.Dont mention the query is doing this and that. Make it sound like a human.
    6.Stricly dont mention about databse or database columns or about the sql query.
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 250
            }
        )
        print(f"Explanation: {response.text.strip()}")
        return response.text.strip()
    except Exception as e:
        return "Error in generating explanation"

#The detect_visualization_request() function determines if the user's question or SQL query implies a need for data visualization. It analyzes the query, dataset, language, and chat history to suggest an appropriate visualization type, such as a bar chart or line graph.
def detect_visualization_request(question,sql,df,lang,chat_history):
    """Detect if the user is requesting a visualization and what type using Gemini model"""
    
    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    # Define the prompt to send to the model
    prompt = f"""
    Chat history: {chat_history[-10:] if chat_history else 'No previous context'}
    The user has asked the following question: "{question}" in {lang} language.
    Generated SQL: {sql} 
    Output of the query:{df}
    Based on the question and chat history, detect if the user is requesting a specific type of visualization (e.g., bar chart, line graph, pie chart, scatter plot, etc.). If so, return the type of visualization (e.g., 'bar', 'line', 'pie', 'scatter'). If no specific visualization is requested, return NULL.
    Even if the user specifies just "plot a graph" or "visualize the data"or "draw a diagram" (any general visualization terms)[must return a visualization for these kinds of questions], use the {sql} and {df} to generate the appropriate visualization type.
    Note: 
    1.The output should be a single word or phrase indicating the type of visualization requested.
    2.Also if the user is asking for a specific type of visualization, return that type.
    3.If the question dosen't require any visualization return NULL (no ouput).
    """
    
    try:
        # Generate a response from the model
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 50
            }
        )
        # Get the model's response
        visualization_type = response.text.strip()
        
        # If the model detects a visualization type, return it
        print(f"Detected visualization type: {visualization_type}")

        if visualization_type!="NULL":
            return visualization_type.lower()
        else:
            return None
    except Exception as e:
        # Handle any errors from the model
        print(f"Error while generating visualization request: {e}")
        return None

#The create_visualization() function generates a data visualization based on the user's question and the DataFrame. It uses the specified visualization type (e.g., bar chart, line chart) to present the data in an informative way.
def create_visualization(question,df, viz_type):
    """Create the visualization based on the type and data using code generated by Gemini"""
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare a prompt to send to Gemini
        prompt = f"""
        I have a dataframe with the following columns:
        {df.columns.to_list()}
        
        Question from User: {question}
        Output of the query: {df}
        The user has requested a {viz_type} visualization.
        Please generate the appropriate Plotly code for this visualization in Python using the dataframe.Also meet the user requirements.

        Note:
        Create appropriate titles, labels, and legends for the visualization.
        Don't put backticks or any other markdown in the code.
        Don't put python word at the front of the code.
        At the end don't print or return fig.
        Don't use functions like def plot_oee_trends(df) or def plot_cycle_time(df). Give the code directly.
        At the end of the code, ensure that a figure object named 'fig' is created.
        """
        
        # Send the prompt to Gemini and get the generated code
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2000
            }
        )
        
        # Get the generated code
        code = response.text.strip()
        
        print(f"visualization: {code}")
        if code:
                # Remove backticks and python from the generated code
                code = code.replace("```python", "").replace("```", "").strip()    
                # Execute the cleaned-up code to create the visualization 
                exec_globals = {
                'df': df,  # Pass the dataframe here
                'px': px  # Assuming 'px' refers to Plotly Express
                }
                exec(code, exec_globals)  # Execute the generated code
                # Return the figure from the local scope after execution
                return exec_globals.get('fig', None) # Assuming 'fig' is defined in the generated code
        
        # If no valid code returned, return an error message visualization
        fig = px.scatter()  # Empty plot as fallback
        fig.add_annotation(
            text="Could not generate the requested visualization.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Visualization Not Available",
            height=500,
            template='plotly_white'
        )
        return fig

    except Exception as e:
        print(f"Error creating visualization: {e}")
        fig = px.scatter()  # Empty plot as fallback
        fig.add_annotation(
            text="Error in visualization creation",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Visualization Error",
            height=500,
            template='plotly_white'
        )
        return fig

#The format_response() function handles the overall process of generating the explanation , detecting visualization requests, and generating the visualization.
def format_response(data, columns, question_type,chat_history,original_question,sql,lang,translated_speech):
    """Format the response in a more conversational way"""
    try:        
        df = pd.DataFrame(data, columns=columns)
        df = df.loc[:, ~df.columns.duplicated()]
        explanation = generate_explanation(df, original_question,sql,lang,chat_history)
        # If query is about machine IP address
        if 'IPAddress' in df.columns and 'MacID' in question_type:
            machine_id = question_type.get('machine_id', '')
            return f"The IP address of {machine_id} is {df.iloc[0]['IPAddress']}"
        elif question_type == 'count':
            return f"I found {len(df)} records matching your query."
        else:
           viz_type = detect_visualization_request(original_question,sql,df,lang,chat_history)
           response = {
                "explanation": explanation,
                "dataframe": df,
                "visualization": None
            }
           fig=None
           if viz_type:
                fig = create_visualization(original_question,df, viz_type)
                if fig:
                    response["visualization"] = fig
           log_query(original_question,lang,translated_speech,sql,explanation,viz_type,fig)
        
           return response
    except Exception:
        if data==3024:
            return "Query maximum time limit reached"
        return "An error occurred while formatting the response."

#The reset_chat() function clears the chat history and resets any stored context, allowing for a fresh conversation without previous interactions influencing responses.
def reset_chat():
    """Reset all chat-related session state variables"""
    st.session_state.chat_history = []
    st.session_state.context = {
        'last_query_type': None,
        'last_entities': {},
        'last_sql': None
    }
    st.session_state.processing = False
    st.session_state.chart_counter = 0
    if 'current_share_id' in st.session_state:
        del st.session_state.current_share_id
    st.session_state.df_size = 0
    st.session_state.dialog_open = -1

#The get_language_options() function retrieves the available language options for user interaction. It returns a list of supported languages for responses, translations, or interface customization.
def get_language_options():
    return {
        'English': 'en-US',
        'Hindi': 'hi-IN',
        'Spanish': 'es-ES',
        'French': 'fr-FR',
        'German': 'de-DE',
        'Italian': 'it-IT',
        'Japanese': 'ja-JP',
        'Korean': 'ko-KR',
        'Chinese': 'zh-CN',
        'Russian': 'ru-RU',
        'Portuguese': 'pt-BR',
        'Arabic': 'ar-SA',
        'Dutch': 'nl-NL',
        'Bengali': 'bn-IN',
        'Turkish': 'tr-TR',
        'Tamil': 'ta-IN',
        'Telugu': 'te-IN',
        'Kannada': 'kn-IN',
        'Malayalam': 'ml-IN',
        'Marathi': 'mr-IN'
    }

#The get_source_language_code() function maps a given language code to its corresponding source language code. It ensures compatibility with translation or localization services by returning the correct language format.
def get_source_language_code(lang_code):
    lang_mapping = {
        'en-US': 'en', 'hi-IN': 'hi', 'es-ES': 'es', 'fr-FR': 'fr', 
        'de-DE': 'de', 'it-IT': 'it', 'ja-JP': 'ja', 'ko-KR': 'ko',
        'zh-CN': 'zh-CN', 'ru-RU': 'ru', 'pt-BR': 'pt', 'ar-SA': 'ar',
        'nl-NL': 'nl', 'bn-IN': 'bn', 'tr-TR': 'tr', 'ta-IN': 'ta',
        'te-IN': 'te', 'kn-IN': 'kn', 'ml-IN': 'ml', 'mr-IN': 'mr'
    }
    return lang_mapping.get(lang_code, 'auto')

#The translate_to_english() function translates the given text from the specified source language to English. It helps in processing multilingual inputs for standardization and further analysis.
def translate_to_english(text, source_lang):
    if not text:
        return ""

    try:
        source_lang_code = get_source_language_code(source_lang)
        if source_lang_code == 'en':
            return text

        translator = GoogleTranslator(source=source_lang_code, target='en')
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return f"Translation failed for: {text}"

#The get_unique_message_id() function generates a unique identifier for a message. This ensures proper tracking and referencing of messages in a conversation or logging system.
def get_unique_message_id():
    st.session_state.unique_copy_id += 1
    return st.session_state.unique_copy_id   

#Flask for TTS
app = Flask(__name__)
CORS(app) 
app.logger.setLevel(logging.INFO)

TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

class VoiceManager:
    def __init__(self):
        self.voices = {}
        self.voice_lock = threading.Lock()
        self.initialized = False
        self.initializing = False
    
    async def initialize_voices(self):
        """Initialize voice list from Edge TTS"""
        if self.initialized or self.initializing:
            return self.initialized
        
        try:
            self.initializing = True
            voices = await edge_tts.list_voices()
            for voice in voices:
                locale = voice["Locale"]
                if locale not in self.voices:
                    self.voices[locale] = []
                self.voices[locale].append({
                    "name": voice["Name"],
                    "gender": voice["Gender"],
                    "is_neural": "Neural" in voice["Name"]
                })
            self.initialized = True
            app.logger.info("Voices initialized successfully")
            return True
        except Exception as e:
            app.logger.error(f"Voice initialization error: {e}")
            return False
        finally:
            self.initializing = False

    def get_best_voice(self, locale):
        if not self.initialized:
            return "en-US-JennyNeural"
            
        if locale not in self.voices:
            locale = "en-US"
        
        available_voices = self.voices.get(locale, [])
        neural_voices = [v for v in available_voices if v["is_neural"]]
        if neural_voices:
            return neural_voices[0]["name"]
        return available_voices[0]["name"] if available_voices else "en-US-JennyNeural"

voice_manager = VoiceManager()

def async_route(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

@app.route('/initialize', methods=['POST'])
@async_route
async def initialize():
    """Separate endpoint for voice initialization"""
    app.logger.info("Received initialization request")
    if voice_manager.initialized:
        app.logger.info("Voice manager already initialized")
        return jsonify({"status": "success", "message": "Already initialized"})
    
    try:
        app.logger.info("Attempting to initialize voices...")
        success = await voice_manager.initialize_voices()
        if success:
            app.logger.info("Voice initialization completed successfully")
            return jsonify({"status": "success", "message": "Initialization complete"})
        else:
            app.logger.error("Voice initialization failed")
            return jsonify({"status": "error", "message": "Initialization failed"}), 500
    except Exception as e:
        app.logger.error(f"Exception during initialization: {str(e)}")
        return jsonify({"status": "error", "message": f"Initialization error: {str(e)}"}), 500

@app.route('/generate_audio', methods=['GET'])
@async_route
async def generate_audio():
    """Generate and return speech audio"""
    try:
        text = request.args.get('text', '')
        locale = request.args.get('lang', 'en-US')
        message_id = request.args.get('message_id', '')

        if not text:
            return "No text provided", 400

        # Ensure voices are initialized
        if not voice_manager.initialized:
            success = await voice_manager.initialize_voices()
            if not success:
                return "Voice initialization failed", 500

        voice_name = voice_manager.get_best_voice(locale)
        timestamp = int(time.time() * 1000)
        output_file = os.path.join(TEMP_DIR, f"speech_{message_id}_{timestamp}.mp3")
        
        try:
            communicate = edge_tts.Communicate(text, voice_name)
            await communicate.save(output_file)
            
            def cleanup():
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except Exception as e:
                    app.logger.error(f"Cleanup error: {e}")
            
            response = send_file(output_file, mimetype="audio/mpeg")
            response.call_on_close(cleanup)
            return response
            
        except Exception as e:
            app.logger.error(f"Speech generation error: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            return str(e), 500

    except Exception as e:
        app.logger.error(f"Request processing error: {e}")
        return str(e), 500
    
TTS_FLASK_PORT = os.getenv("PORT2", 8509)
def run_flask():
    try:
        serve(app,host=os.getenv("HOST",'0.0.0.0'), port=TTS_FLASK_PORT)
    except Exception as e:
        app.logger.error(f"Server error: {e}")

if not any(f":{TTS_FLASK_PORT}" in line for line in os.popen("netstat -ano").readlines()):
    threading.Thread(target=run_flask, daemon=True).start()

#The copy_audio_script() function generates an interactive audio script. It includes text, speaker icons, GIF animations, and feedback icons to enhance user engagement. The function ensures proper handling of audio playback and language-specific rendering.
def copy_audio_script(text_to_copy, gif_base64, speaker_idle_icon_base64, speaker_playing_icon_base64,thumbs_up_icon_base64,thumbs_down_icon_base64, unique_copy_id, language_code):
    """Copy-to-clipboard and on-demand TTS functionality"""
    
    encoded_text = urllib.parse.quote(text_to_copy)
    audio_api_host = os.getenv("AUDIO_API_HOST", "http://localhost:8509")
    audio_url = f"{audio_api_host}/generate_audio?text={encoded_text}&lang={language_code}&message_id={unique_copy_id}"
    escaped_text = text_to_copy.replace('`', '\\`').replace('${', '\\${')

    copy_script = f"""
    <style>
        *{{
        padding-left:0px;
        margin-left:0px;
        }}
        .btn-container {{
            display: flex;
            align-items: center;
            gap: 0px;
        }}
        .btn {{
            background-color: transparent;
            border: none;
            cursor: pointer;
            padding: 0px;
            position: relative;
        }}
        .btn img {{
            width: 40px;
            height: 40px;
            transition: transform 0.3s ease;
        }}
        .btn .tooltip {{
        visibility: hidden;
        width: auto;
        background-color: #31333F;
        color: #fff;
        text-align: center;
        padding: 3px;
        border-radius: 5px;
        position: absolute;
        bottom: 80%;
        left: 50%;
        transform: translateX(-50%);
        font-size: 10px;
        opacity: 0;
        transition: opacity 0.3s;
        white-space: nowrap;
        }}
        .btn:hover .tooltip {{
            visibility: visible;
            opacity: 1;
        }}
        .btn .tooltip.visible {{
            visibility: visible;
            opacity: 1;
        }}
        .btn:active img {{
            transform: scale(0.7);
        }}
        .loading-spinner {{
            position: absolute;
            top: 43%;
            left: 50%;
            width: 12px;
            height: 12px;
            border: 2px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: none;
        }}
        @keyframes spin {{
            0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
            100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
        }}
        .error-message {{
            font-family:sans-serif;
            color: red;
            font-size: 12px;
            display: none;
            margin-left: 10px;
        }}
        .faded {{
            opacity: 0.3;
        }}
    </style>
    
    <div class="btn-container" id="container-{unique_copy_id}">
        <button class="btn" onclick="void(0)" id="copy-btn-{unique_copy_id}">
            <span class="tooltip" id="copy-tooltip-{unique_copy_id}">Copy</span>
            <img id="copy-icon-{unique_copy_id}" src="data:image/gif;base64,{gif_base64}" alt="Copy">
        </button>
        <button class="btn" onclick="void(0)" id="thumbs-up-btn-{unique_copy_id}">
            <span class="tooltip" id="thumbs-up-tooltip-{unique_copy_id}">Good Response</span>
            <img id="thumbs-up-icon-{unique_copy_id}" src="data:image/gif;base64,{thumbs_up_icon_base64}" alt="Thumbs up">
        </button>
        <button class="btn" onclick="void(0)" id="thumbs-down-btn-{unique_copy_id}">
            <span class="tooltip" id="thumbs-down-tooltip-{unique_copy_id}">Bad Response</span>
            <img id="thumbs-down-icon-{unique_copy_id}" src="data:image/gif;base64,{thumbs_down_icon_base64}" alt="Thumbs down">
        </button>
        <button class="btn" onclick="void(0)" id="speaker-btn-{unique_copy_id}">
            <span class="tooltip">Read aloud</span>
            <img id="speaker-icon-{unique_copy_id}" src="data:image/gif;base64,{speaker_idle_icon_base64}" alt="Speaker">
            <div id="loading-spinner-{unique_copy_id}" class="loading-spinner"></div>
        </button>
        <div id="error-{unique_copy_id}" class="error-message"></div>
    </div>

    <audio id="audio-{unique_copy_id}" preload="none"></audio>

    <script>
        (function() {{
            const container = document.getElementById('container-{unique_copy_id}');
            if (!container) return;

            const copyBtn = document.getElementById('copy-btn-{unique_copy_id}');
            const copyTooltip = document.getElementById('copy-tooltip-{unique_copy_id}');
            const speakerBtn = document.getElementById('speaker-btn-{unique_copy_id}');
            const thumbsUpBtn = document.getElementById('thumbs-up-btn-{unique_copy_id}');
            const thumbsDownBtn = document.getElementById('thumbs-down-btn-{unique_copy_id}');
            const thumbsUpTooltip = document.getElementById('thumbs-up-tooltip-{unique_copy_id}');
            const thumbsDownTooltip = document.getElementById('thumbs-down-tooltip-{unique_copy_id}');
            const thumbsUpIcon = document.getElementById('thumbs-up-icon-{unique_copy_id}');
            const thumbsDownIcon = document.getElementById('thumbs-down-icon-{unique_copy_id}');
            const audio = document.getElementById('audio-{unique_copy_id}');
            const speakerIcon = document.getElementById('speaker-icon-{unique_copy_id}');
            const errorDiv = document.getElementById('error-{unique_copy_id}');
            const loadingSpinner = document.getElementById('loading-spinner-{unique_copy_id}');
            
            // Global initialization state
            if (typeof window.voicesInitialized === 'undefined') {{
                window.voicesInitialized = false;
                window.initializingVoices = false;
            }}

            copyBtn.addEventListener('click', function() {{
                navigator.clipboard.writeText(`{escaped_text}`).then(() => {{
                    const icon = document.getElementById('copy-icon-{unique_copy_id}');
                    icon.style.transform = 'scale(1.2)';
                    copyTooltip.textContent = 'Copied';
                    copyTooltip.classList.add('visible');
                    setTimeout(() => {{
                        icon.style.transform = 'scale(1)';
                        setTimeout(() => {{
                            copyTooltip.classList.remove('visible');
                            copyTooltip.textContent = 'Copy';
                        }}, 2000);
                    }}, 300);
                }}).catch(err => {{
                    console.error('Clipboard copy failed:', err);
                    showError('Failed to copy text');
                }});
            }});

            thumbsUpBtn.addEventListener('click', function() {{
                // First, reset both buttons
                thumbsUpIcon.classList.remove('faded');
                thumbsDownIcon.classList.remove('faded');
                
                // Check if this button is already selected
                if (thumbsUpTooltip.textContent === 'Thanks for your response') {{
                    // Reset if already selected
                    thumbsUpTooltip.textContent = 'Good Response';
                    thumbsUpTooltip.classList.remove('visible');
                }} else {{
                    // Select this button and fade the other
                    thumbsDownIcon.classList.add('faded');
                    thumbsUpTooltip.textContent = 'Thanks for your response';
                    thumbsUpTooltip.classList.add('visible');
                    
                    // Reset the other button
                    thumbsDownTooltip.textContent = 'Bad Response';
                    thumbsDownTooltip.classList.remove('visible');
                    
                    // Show tooltip for 2 seconds
                    setTimeout(() => {{
                        thumbsUpTooltip.classList.remove('visible');
                    }}, 2000);
                }}
            }});

            thumbsDownBtn.addEventListener('click', function() {{
                // First, reset both buttons
                thumbsUpIcon.classList.remove('faded');
                thumbsDownIcon.classList.remove('faded');
                
                // Check if this button is already selected
                if (thumbsDownTooltip.textContent === 'Thanks for your response') {{
                    // Reset if already selected
                    thumbsDownTooltip.textContent = 'Bad Response';
                    thumbsDownTooltip.classList.remove('visible');
                }} else {{
                    // Select this button and fade the other
                    thumbsUpIcon.classList.add('faded');
                    thumbsDownTooltip.textContent = 'Thanks for your response';
                    thumbsDownTooltip.classList.add('visible');
                    
                    // Reset the other button
                    thumbsUpTooltip.textContent = 'Good Response';
                    thumbsUpTooltip.classList.remove('visible');
                    
                    // Show tooltip for 2 seconds
                    setTimeout(() => {{
                        thumbsDownTooltip.classList.remove('visible');
                    }}, 2000);
                }}
            }});

            function showError(message) {{
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
                setTimeout(() => {{
                    errorDiv.style.display = 'none';
                }}, 3000);
            }}

            async function initializeVoices() {{
                if (window.voicesInitialized) return true;
                if (window.initializingVoices) return false;
                
                window.initializingVoices = true;
                loadingSpinner.style.display = 'block';
                
                try {{
                    const response = await fetch('{audio_api_host}/initialize', {{
                        method: 'POST'
                    }});
                    
                    if (!response.ok) throw new Error('Initialization failed');
                    
                    const data = await response.json();
                    if (data.status === 'success') {{
                        window.voicesInitialized = true;
                        return true;
                    }}
                    throw new Error(data.message);
                }} catch (err) {{
                    console.error('Voice initialization error:', err);
                    showError('Failed to initialize voices');
                    return false;
                }} finally {{
                    window.initializingVoices = false;
                    loadingSpinner.style.display = 'none';
                }}
            }}

            speakerBtn.addEventListener('click', async function() {{
                if (!window.voicesInitialized && !window.initializingVoices) {{
                    const success = await initializeVoices();
                    if (!success) return;
                }}

                if (audio.src === '') {{
                    audio.src = "{audio_url}";
                }}

                if (audio.paused) {{
                    loadingSpinner.style.display = 'block';
                    audio.play().catch(err => {{
                        console.error('Audio playback error:', err);
                        showError('Failed to play audio');
                        speakerIcon.src = "data:image/gif;base64,{speaker_idle_icon_base64}";
                    }}).finally(() => {{
                        loadingSpinner.style.display = 'none';
                    }});
                    speakerIcon.src = "data:image/gif;base64,{speaker_playing_icon_base64}";
                }} else {{
                    audio.pause();
                    audio.currentTime = 0;
                    speakerIcon.src = "data:image/gif;base64,{speaker_idle_icon_base64}";
                }}
            }});

            audio.addEventListener('ended', function() {{
                speakerIcon.src = "data:image/gif;base64,{speaker_idle_icon_base64}";
            }});

            audio.addEventListener('error', function(e) {{
                speakerIcon.src = "data:image/gif;base64,{speaker_idle_icon_base64}";
                loadingSpinner.style.display = 'none';
                console.error('Audio error:', e);
                showError('Failed to load audio');
            }});
        }})();
    </script>
    """
    components.html(copy_script, height=42)

#he copy_to_clipboard_script() function generates a JavaScript-based script to copy text to the clipboard. It also displays an animation (GIF) to indicate successful copying, using a unique identifier for tracking.
def copy_to_clipboard_script(text_to_copy, gif_base64, unique_copy_id):
    copy_script = f"""
    <style>
        body {{
            margin: 0px;
            padding: 0px;
        }}
        .copy-container {{
            display: flex;
            align-items: center;
        }}
        .copy-message {{
        font-family:sans-serif; 
            margin-left: 0px;
            font-size: 14px;
            color: gray;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
    </style>
    <div class="copy-container">
        <button onclick="copyToClipboard_{unique_copy_id}()" style="background-color: transparent; border: none; font-size: 14px; cursor: pointer;">
            <img id="copy-icon-{unique_copy_id}" src="data:image/gif;base64,{gif_base64}" alt="Copy Icon" style="width: 40px; height: 40px; transition: transform 0.3s ease;" />
        </button>
        <span id="copy-message-{unique_copy_id}" class="copy-message">Copied!</span>
    </div>
    <script>
        function copyToClipboard_{unique_copy_id}() {{
            navigator.clipboard.writeText(`{text_to_copy}`).then(() => {{
                document.getElementById('copy-icon-{unique_copy_id}').style.transform = 'scale(1.2)';
                document.getElementById('copy-message-{unique_copy_id}').style.opacity = '1';
                setTimeout(() => {{
                    document.getElementById('copy-icon-{unique_copy_id}').style.transform = 'scale(1)';
                    document.getElementById('copy-message-{unique_copy_id}').style.opacity = '0';
                }}, 1000);
            }}).catch(err => {{
                console.error('Clipboard copy failed:', err);
            }});
        }}
    </script>
    """
    components.html(copy_script, height=42)

#The export_chat_history() function exports the chat history in the specified language. It formats the conversation into a structured file HTML.
def export_chat_history(chat_history, language):
    """
    Export chat history to a polished HTML format with modern design, professional styling, and properly positioned user/bot icons.
    """
    html_content = f"""
    <html>
    <head>
        <title>Chat Export ({language})</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background-color: #f5f7fb;
                margin: 0;
                padding: 40px 20px;
                line-height: 1.6;
            }}
            .chat-container {{
                max-width: 900px;
                margin: 0 auto;
                background-color: #ffffff;
                padding: 32px;
                border-radius: 12px;
                box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
            }}
            h1 {{
                font-size: 28px;
                color: #1a1a1a;
                margin-bottom: 24px;
                font-weight: 600;
                letter-spacing: -0.5px;
            }}
            .export-info {{
                color: #666;
                font-size: 14px;
                margin-bottom: 32px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eaecef;
            }}
            .message {{
                margin-bottom: 24px;
                display: flex;
                gap: 16px;
                align-items: flex-start;
            }}
            .message .icon {{
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                flex-shrink: 0;
                color: white;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .message-content {{
                flex: 1;
                padding: 16px 20px;
                border-radius: 12px;
                position: relative;
                max-width:800px;
            }}
            .user .message-content {{
                background-color: #f8f9fa;
                border: 1px solid #eaecef;
            }}
            .assistant .message-content {{
                background-color: #f0f7ff;
                border: 1px solid #e1ecf7;
            }}
            .timestamp {{
                font-size: 12px;
                color: #8c8c8c;
                margin-top: 8px;
                display: block;
            }}
            .scrollable-table {{
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #eaecef;
                margin: 16px 0;
                background: #ffffff;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                color: #333;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #1a1a1a;
                position: sticky;
                top: 0;
                z-index: 1;
                padding: 12px 16px;
                border-bottom: 2px solid #eaecef;
            }}
            td {{
                padding: 12px 16px;
                border-bottom: 1px solid #eaecef;
            }}
            tr:hover {{
                background-color:rgb(194, 196, 198);
            }}
            code {{
                font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                background: #f3f4f6;
                padding: 2px 4px;
                border-radius: 4px;
                font-size: 0.9em;
            }}
            pre {{
                background: #f3f4f6;
                padding: 16px;
                border-radius: 8px;
                overflow-x: auto;
                margin: 16px 0;
            }}
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>Chat Export ({language})</h1>
            <div class="export-info">
                <strong>Export Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
    """

    # Modern emoji for User and Assistant
    user_icon = "👤"  # User Icon
    assistant_icon = "🤖"  # Assistant Bot Icon

    for message in chat_history:
        timestamp = f"<span class='timestamp'>{datetime.fromtimestamp(message['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</span>" if 'timestamp' in message else ''
        
        if message['role'] == 'user':
            html_content += f"""
            <div class="message user">
                <div class="icon" style="background: linear-gradient(135deg, #4CAF50, #45a049);">{user_icon}</div>
                <div class="message-content">
                    {message['content']}
                    {timestamp}
                </div>
            </div>
            """
        elif message['role'] == 'assistant':
            html_content += f"""
            <div class="message assistant">
                <div class="icon" style="background: linear-gradient(135deg, #2196F3, #1976D2);">{assistant_icon}</div>
                <div class="message-content">
            """
            if isinstance(message['content'], dict):
                html_content += f"<p>{message['content'].get('explanation', 'No explanation')}</p>\n"
                
                if 'dataframe' in message['content']:
                    df = message["content"]["dataframe"]
                    if df.shape[1] > 1:
                        html_content += f"""
                            <div class="scrollable-table">
                                {message['content']['dataframe'].to_html(classes=['table'])}
                            </div>
                        """
                if 'visualization' in message['content'] and message['content']['visualization']:
                    fig_html = pio.to_html(
                        message['content']['visualization'].update_layout(template="plotly"),
                        full_html=False,
                        include_plotlyjs=False,
                        config={
                            "displayModeBar": "hover",
                            "showLink": False,
                            "responsive": True
                        }
                    )
                    html_content += f"""
                    <div style="margin: 20px 0;">
                        {fig_html}
                    </div>
                    """
            else:
                html_content += f"{message['content']}"
            
            html_content += f"""
                    {timestamp}
                </div>
            </div>
            """
    
    html_content += "</div></body></html>"
    return html_content

#The download_chat_button() function creates a downloadable button that allows users to export and download their chat history in the specified language.    
def download_chat_button(chat_history, language):
    """
    Create a download button for chat export.
    
    Args:
    chat_history (list): List of chat messages
    language (str): Language of the chat
    
    Returns:
    None (triggers file download)
    """
    if chat_history:

        if 'download_clicked' not in st.session_state:
            st.session_state.download_clicked = False

        html_content = export_chat_history(chat_history, language)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_file.write(html_content.encode('utf-8'))
            temp_file_path = temp_file.name
        
        # Read the file and provide download
        with open(temp_file_path, 'rb') as file:
            st.download_button(
                label="Download Chat",
                data=file,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_chat_btn",
                on_click=lambda:( set_download_clicked(),stop_processing()),
                disabled=st.session_state.processing
            )
        def set_download_clicked():
            st.session_state.download_clicked = True

        if st.session_state.download_clicked:
                st.toast("Chat history downloaded successfully!", icon="✅")
                st.session_state.download_clicked = False
        # Remove the temporary file
        os.unlink(temp_file_path)
        
    else:
        st.warning("No chat history to export.")

#The get_db_connection() function establishes and returns a connection to the database. It ensures proper configuration, authentication, and error handling for reliable database access.
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
#Flask for sharing chat via link
app = Flask(__name__)

def get_or_create_share_id():
    """Get existing share ID for current chat session or create new one"""
    if 'current_share_id' not in st.session_state:
        st.session_state.current_share_id = str(uuid.uuid4())
    return st.session_state.current_share_id

# Replace your existing save_shared_chat function with this updated version:
def save_shared_chat(html_content, username, mail_id):
    """Save or update chat HTML in database"""
    share_id = get_or_create_share_id()  # Get existing or new share ID
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if share_id exists
    cursor.execute(
        "SELECT COUNT(*) FROM shared_chats WHERE link_id = %s",
        (share_id,)
    )
    exists = cursor.fetchone()[0] > 0
    
    if exists:
        # Update existing record
        cursor.execute(
            "UPDATE shared_chats SET html_content = %s WHERE link_id = %s",
            (html_content, share_id)
        )
    else:
        # Create new record
        cursor.execute(
            "INSERT INTO shared_chats (link_id, created_at, user_name, mail_id, html_content) VALUES (%s, NOW(), %s, %s, %s)",
            (share_id, username, mail_id, html_content)
        )
    
    conn.commit()
    conn.close()
    return share_id

@app.route('/shared-chat/<link_id>')

def view_shared_chat(link_id):
    """Serve the shared chat HTML"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT html_content FROM shared_chats WHERE link_id = %s",
        (link_id,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]  # Return HTML content
    base64_image = get_base64_image("404-page_not_found.png")
    
    # If chat is not found, return an HTML page with an embedded Base64 image
    return Response(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Page Not Found</title>
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f8f9fa;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            @media (max-width: 512px) {{
                img {{
                    display: block;
                    margin: 0 auto;
                    width: 80%;
                }}
            }}
        </style>
    </head>
    <body>
        <img src="data:image/png;base64,{base64_image}" alt="404 Not Found">
    </body>
    </html>
    """, mimetype="text/html"), 404

def start_flask_server():
    """Start Flask server in a separate thread"""
    serve(app,host=os.getenv("HOST",'0.0.0.0'), port=os.getenv("PORT", 8510))

# Initialize Flask server in a separate thread when the app starts
def init_app():
    if 'flask_server_started' not in st.session_state:
        # init_db()
        PORT = os.getenv("PORT", 8510)
        if not any(f":{PORT}" in line for line in os.popen("netstat -ano").readlines()):
            thread = threading.Thread(target=start_flask_server, daemon=True)
            thread.start()
            st.session_state.flask_server_started = True 

@st.dialog("Chat Link")
def show_share_dialog(share_link):
    st.code(share_link, language="html")
    unique_copy_id = get_unique_message_id()  
    gif_base64 = get_base64_image("copy.png")
    copy_to_clipboard_script(share_link, gif_base64,unique_copy_id)
    st.write("Share this link with others to view the chat.") 

#The log_query() function records details of a user's query. It logs the question, language, translated speech, SQL query, explanation, visualization type, and figure, along with an optional timestamp for tracking only for the coorect output.
def log_query(question,lang, translated_speech, sql_query, explanation, viz_type,figure, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "language": lang,
        "translated": translated_speech,
        "sql_query": sql_query,
        "explanation": explanation,
        "visualization_type": viz_type
    }
    
    # Add figure data if available
    if figure is not None:
        try:
            # Convert the figure to a dictionary representation
            figure_dict = figure.to_dict()
            
            # Convert numpy arrays to lists and datetime objects to strings
            for trace in figure_dict['data']:
                if 'x' in trace:
                    trace['x'] = [x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, 'strftime') else 
                                  x.tolist() if isinstance(x, np.ndarray) else x for x in trace['x']]
                if 'y' in trace:
                    trace['y'] = [float(y) if isinstance(y, (np.floating, np.integer)) 
                                  else y.tolist() if isinstance(y, np.ndarray) else y for y in trace['y']]
            
            log_entry["visualization_data"] = json.dumps(figure_dict, separators=(',', ':'), default=str)
            
        except Exception as e:
            print(f"Error processing figure data: {e}")
            # If there's an error processing the figure, still log the rest of the data
            log_entry["visualization_data"] = "Error: Could not serialize figure data"
    
    log_file = "query_logs.json"
    
    try:
        # Read existing logs if file exists
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new log entry
        logs.append(log_entry)
        
        # Write updated logs back to file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error logging query: {e}")

#The log_error() function records errors that occur during query execution. It logs the question, translated speech, SQL query, error type, and exception details, along with an optional timestamp for debugging and analysis for errors alone.
def log_error(question,translated_speech,sql,error_type,e,timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "translated":translated_speech,
        "sql": sql,
        "error_type": error_type,
        "error_message": str(e)
    }
    
    log_file = "error_logs.json"
    
    try:
        # Read existing logs if file exists
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new log entry
        logs.append(log_entry)
        
        # Write updated logs back to file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error logging error: {e}")

#Cookies initialization
cookies = EncryptedCookieManager(
    prefix="ai_buddy_",  # Set a prefix for your cookies
    password=os.getenv("COOKIE_PASSWORD") # Set a password to encrypt your cookies
)

# Wait until cookies are ready
if not cookies.ready():
    st.stop()

#The db_user_credentials() function connects to a MySQL database and executes a given SQL query. It returns the fetched rows and column names, handling exceptions and errors gracefully. Used for authenctication.
def db_user_credentials(sql, db):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=db
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [i[0] for i in cur.description]
        conn.close()
        return rows, columns
    except Exception as e:
        error_type = type(e).__name__  # Get the error type
        print(f"[{error_type}] Database query error: {e}")
        return None, None

#The get_user_credentials() function retrieves the stored credentials for a given username. It is used for authentication and user verification.
def get_user_credentials(username):
    """Fetch the hashed password for a given username from the database."""
    sql = f"SELECT hashed_password,full_name FROM user_credentials WHERE email_address = '{username}'"
    rows, columns = db_user_credentials(sql, db=os.getenv("MYSQL_DB"))
    if rows and len(rows) > 0:
        return rows[0][0], rows[0][1]  # The first column of the first row contains the hashed password
    return None,None

#The check_password() function verifies whether the provided password matches the stored hashed password. It is used for authentication by securely comparing user credentials.
def check_password(provided_password, stored_hashed_password):
    """Validate the provided password against the stored hashed password."""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hashed_password.encode('utf-8'))

#The check_auth() function verifies whether the user is authenticated. It checks for valid credentials or session tokens to grant or restrict access to protected resources.
def check_auth():
    """Check username and password for access."""
    # Check if the user is logged in via cookies
    if cookies.get("logged_in") == "true":
        st.session_state["logged_in"] = True
        st.session_state["current_user"] = cookies.get("username")
        st.session_state["full_name"] = cookies.get("full_name")
        return True

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        # Center the login form using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.header("Welcome")
            with st.form("login_form", clear_on_submit=False):
                st.subheader("Login")
                username = st.text_input("Username", placeholder="Enter your email", autocomplete="username")
                password = st.text_input("Password", placeholder="Enter your password", type="password", autocomplete="current-password")
                
                # Center the login button
                col1, col2, col3 = st.columns([1.2, 1, 1])
                with col2:
                    login_button = st.form_submit_button("Login")

                if login_button:
                    stored_hashed_password,full_name = get_user_credentials(username)
                    if stored_hashed_password and check_password(password, stored_hashed_password):
                        st.session_state["logged_in"] = True
                        st.session_state["current_user"] = username
                        st.session_state["full_name"] = full_name
                        # Save login state in cookies
                        cookies["logged_in"] = "true"
                        cookies["username"] = username
                        cookies["full_name"] = full_name
                        cookies.save()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        return False
    else:
        return True

#The get_base64_image(file_path) function reads an image file from the specified path and converts it into a Base64-encoded string. This allows images to be embedded in web pages or transmitted as text.    
def get_base64_image(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()
    
#The stop_processing() function halts any ongoing processing or computation. It is used to gracefully stop tasks, free resources, or handle user-triggered cancellations.    
def stop_processing():
    st.session_state.stop_requested = True
    st.session_state.processing = False    

#The main() function serves as the entry point of the program. It initializes necessary components, sets up configurations, and runs the core logic of the application.
def main():       
    if not check_auth():
        return
    st.markdown("""
        <style>   
            .st-emotion-cache-ocqkz7{
                position: fixed;
                bottom: 20px;
                width:inherit;
                flex-wrap:nowrap;
                padding:3px;    
                gap: 2px;
            }
            .st-key-logout_button{
                position: fixed;
                bottom: 25px;
            }
        </style>
    """, unsafe_allow_html=True)
    # Create a container for the header and new chat button
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    with st.sidebar:       
        # New Chat button
        if st.session_state.get("full_name"):
            st.write(f"Welcome, {st.session_state['full_name']}!")
            st.markdown("<hr>", unsafe_allow_html=True) 
        if st.button("New Chat", type="primary"):
            reset_chat()
            st.rerun()
        
        # Language Selection Dropdown
        st.subheader("Language")
        language_options = get_language_options()
        selected_language = st.selectbox(
            "Choose Language", 
            list(language_options.keys()), 
            index=0 ,# Default to English
            disabled=st.session_state.processing,
            on_change=stop_processing
        )
        selected_language_code = language_options[selected_language]
        st.subheader("Chat Export")
        if st.session_state.chat_history:
            chat_export_choice = st.radio("Choose export", ("Download", "Share"), label_visibility="collapsed",disabled=st.session_state.processing,on_change=stop_processing)
        
            # If "Download" is selected
            if chat_export_choice == "Download":
                # Display the Download button
                if st.session_state.chat_history:
                    download_chat_button(st.session_state.chat_history, selected_language)
        
            # If "Share" is selected
            elif chat_export_choice == "Share":
                # Display the Share button
                if st.button("Share Chat", key="share_chat_button", disabled=st.session_state.processing,on_click=stop_processing):
                    if not st.session_state.get("logged_in", False):
                        st.error("Please log in to share chats.")
                    else:
                        html_content = export_chat_history(st.session_state.chat_history, selected_language)
            
                        # Get user info
                        mail_id = st.session_state.get("current_user")
                        username = st.session_state.get("full_name")
                        
                        # Generate a new share ID if not exists
                        if 'current_share_id' not in st.session_state:
                            st.session_state.current_share_id = str(uuid.uuid4())
                        
                        # Get the share ID
                        share_id = st.session_state.current_share_id
                        
                        # Save to database
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        
                        try:
                            # Check if share_id already exists
                            cursor.execute(
                                "SELECT COUNT(*) FROM shared_chats WHERE link_id = %s",
                                (share_id,)
                            )
                            exists = cursor.fetchone()[0] > 0
                            
                            if exists:
                                # Update existing record
                                cursor.execute(
                                    "UPDATE shared_chats SET html_content = %s WHERE link_id = %s",
                                    (html_content, share_id)
                                )
                            else:
                                # Insert new record
                                cursor.execute(
                                    "INSERT INTO shared_chats (link_id, created_at, user_name, mail_id, html_content) VALUES (%s, NOW(), %s, %s, %s)",
                                    (share_id, username, mail_id, html_content)
                                )
                            
                            conn.commit()
                            
                            # Generate share link and show dialog
                            SHARE_CHAT_URL = os.getenv("SHARE_CHAT_URL", "http://localhost:8510")
                            share_link = f"{SHARE_CHAT_URL}/shared-chat/{share_id}"
                            show_share_dialog(share_link)
                            
                        except Exception as e:
                            st.error(f"Error saving chat: {str(e)}")
                        finally:
                            conn.close()

        else:
            st.warning("No chat history to export.")
        if st.session_state["logged_in"]:
            if st.button("Logout", key="logout_button"):
                cookies["logged_in"] = "false"
                cookies.pop("username", None)
                cookies.save()
                st.session_state["logged_in"] = False
                st.session_state.pop("current_user", None)  # Clear the current user info
                st.session_state.chat_history = []  # Clear chat history
                st.rerun()
        # # Placeholder for future chat history
        # st.subheader("Chat History")
        # st.markdown("*No chats available yet.*") 
        
    # Initialize session states
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'chart_counter' not in st.session_state:
        st.session_state.chart_counter = 0

    if 'context' not in st.session_state:
        st.session_state.context = {
            'last_query_type': None,
            'last_entities': {},
            'last_sql': None
        }
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    # Display chat history and loading state
    if 'displayed_messages' not in st.session_state:
        st.session_state.displayed_messages = set()
    
    if 'typing_message' not in st.session_state:
        st.session_state.typing_message = None

    if 'stop_requested' not in st.session_state:
       st.session_state.stop_requested = False
    
    if 'df_size' not in st.session_state:
        st.session_state.df_size = 0
    
    if 'dialog_open' not in st.session_state:
        st.session_state.dialog_open = -1

# Function to handle stop button click

    for i, message in enumerate(st.session_state.chat_history):
        message_id = f"{i}_{message['role']}_{hash(str(message['content']))}"
        
        with st.chat_message(message["role"]):
            # For user messages, display them immediately
            if message["role"] == "user":
                st.markdown(message["content"])
            # For assistant messages, apply typing effect if not displayed yet
            elif message["role"] == "assistant" and message_id not in st.session_state.displayed_messages:
                st.session_state.typing_message = True
                if isinstance(message["content"], dict):
                    # For dictionary content, create a placeholder for typing effect
                    placeholder = st.empty()
                    full_text = message["content"]["explanation"]
                    
                    # Simulate typing effect for explanation
                    for j in range(len(full_text) + 1):
                        if st.session_state.get("stop_requested", False):
                            break
                        placeholder.markdown(full_text[:j] + "▌")
                        time.sleep(0.01)
                    placeholder.markdown(full_text)
                    st.session_state.stop_requested = False
                    
                    if "visualization" in message["content"] and message["content"]["visualization"] is not None and "dataframe" in message["content"]:
                        tabs=st.tabs(["Visualization", "Dataframe"])
                        with tabs[0]:
                            chart_key = f"chart_{i}_{st.session_state.chart_counter}"
                            st.plotly_chart(
                                message["content"]["visualization"], 
                                use_container_width=True,
                                key=chart_key
                            )
                            st.session_state.chart_counter += 1
                        with tabs[1]:
                            df = message["content"]["dataframe"]
                            if df.shape[1] > 1:
                                st.dataframe(
                                    message["content"]["dataframe"],
                                    use_container_width=True,
                                    hide_index=True,
                                    key=f"df_{i}"
                                )

                    # Display visualization if available
                    elif "visualization" in message["content"] and message["content"]["visualization"] is not None:
                        chart_key = f"chart_{i}_{st.session_state.chart_counter}"
                        st.plotly_chart(
                            message["content"]["visualization"], 
                            use_container_width=True,
                            key=chart_key
                        )
                        st.session_state.chart_counter += 1
                    
                    # Display dataframe
                    elif "dataframe" in message["content"]:
                        df = message["content"]["dataframe"]
                        if df.shape[1] > 1:
                            st.dataframe(
                                message["content"]["dataframe"],
                                use_container_width=True,
                                hide_index=True,
                                key=f"df_{i}"
                            )
                    print ("Dataframe memory",df.memory_usage(deep=True).sum())
                    st.session_state.df_size+=df.memory_usage(deep=True).sum()
                else:
                    # For regular text content, create a placeholder for typing effect
                    placeholder = st.empty()
                    full_text = message["content"]
                    
                    # Simulate typing effect
                    for j in range(len(full_text) + 1):
                        if st.session_state.get("stop_requested", False):
                            break
                        placeholder.markdown(full_text[:j] + "▌")
                        time.sleep(0.01)
                    placeholder.markdown(full_text)
                    st.session_state.stop_requested = False
                
                st.session_state.typing_message = False
                st.session_state.displayed_messages.add(message_id)
                unique_copy_id = get_unique_message_id()
                if isinstance(message["content"], dict):
                    text_to_copy = message["content"]["explanation"]
                else:
                    text_to_copy = message["content"]
                gif_base64 = get_base64_image("copy.png")
                speaker_idle_icon_base64=get_base64_image("speaker.png")
                speaker_playing_icon_base64=get_base64_image("pause.png")
                thumbs_up_icon_base64=get_base64_image("thumbsup.png")
                thumbs_down_icon_base64=get_base64_image("thumbsdown.png")
                copy_audio_script(text_to_copy, gif_base64,speaker_idle_icon_base64,speaker_playing_icon_base64,thumbs_up_icon_base64,thumbs_down_icon_base64,unique_copy_id,selected_language_code)
                total_chat_length = st.session_state.df_size
                print(f'Total dataframe memory: {total_chat_length}')
                if total_chat_length > 1000000:
                    st.session_state.dialog_open += 1
                    if st.session_state.dialog_open % 5 == 0:
                        @st.dialog("Chat Limit Reached")
                        def show_dialog():
                            st.write("The context may not be followed beyond this point. Please start a new chat to continue.")
                            
                            if st.button("New Chat", type="primary", key="dialog_new_chat_button"):
                                reset_chat()
                                st.rerun()
                        show_dialog()
            # For already displayed assistant messages, show them immediately
            else:
                if isinstance(message["content"], dict):
                    st.markdown(message["content"]["explanation"])

                    if "visualization" in message["content"] and message["content"]["visualization"] is not None and "dataframe" in message["content"]:
                        tabs=st.tabs(["Visualization", "Dataframe"])
                        with tabs[0]:
                            chart_key = f"chart_{i}_{st.session_state.chart_counter}"
                            st.plotly_chart(
                                message["content"]["visualization"], 
                                use_container_width=True,
                                key=chart_key
                            )
                            st.session_state.chart_counter += 1
                        with tabs[1]:
                            df = message["content"]["dataframe"]
                            if df.shape[1] > 1:
                                st.dataframe(
                                    message["content"]["dataframe"],
                                    use_container_width=True,
                                    hide_index=True,
                                    key=f"df_{i}"
                                )

                    elif "visualization" in message["content"] and message["content"]["visualization"] is not None:
                        chart_key = f"chart_{i}_{st.session_state.chart_counter}"
                        st.plotly_chart(
                            message["content"]["visualization"], 
                            use_container_width=True,
                            key=chart_key
                        )
                        st.session_state.chart_counter += 1
                    
                    elif "dataframe" in message["content"]:
                        df = message["content"]["dataframe"]
                        if df.shape[1] > 1:
                            st.dataframe(
                                message["content"]["dataframe"],
                                use_container_width=True,
                                hide_index=True,
                                key=f"df_{i}"
                            )
                else:
                    st.markdown(message["content"])
                if isinstance(message["content"], dict):
                    text_to_copy = message["content"]["explanation"]
                else:
                    text_to_copy = message["content"]
                unique_copy_id = get_unique_message_id()
                gif_base64 = get_base64_image("copy.png")
                speaker_idle_icon_base64=get_base64_image("speaker.png")
                speaker_playing_icon_base64=get_base64_image("pause.png")
                thumbs_up_icon_base64=get_base64_image("thumbsup.png")
                thumbs_down_icon_base64=get_base64_image("thumbsdown.png")
                copy_audio_script(text_to_copy, gif_base64,speaker_idle_icon_base64,speaker_playing_icon_base64,thumbs_up_icon_base64,thumbs_down_icon_base64,unique_copy_id,selected_language_code)
    
    # Show loading message if processing
    if st.session_state.processing:
        with st.chat_message("assistant"):
            time.sleep(0.5)
            gif_base64 = get_base64_image("loading.gif")
            st.markdown(
            f"""
                <div style="display: flex;">
                <p style="animation: dim-bright 1.5s infinite;">Analyzing...</p>
                <img src="data:image/gif;base64,{gif_base64}" alt="Processing..." style="width:50px; height:30px; vertical-align: top; margin-top: 0px;"/>
                </div>
                <style>
                @keyframes dim-bright {{
                    0% {{ opacity: 1; }}
                    50% {{ opacity: 0.3; }}
                    100% {{ opacity: 1; }}
                }}
                </style>
            """,
            unsafe_allow_html=True,)
            time.sleep(0.1)
    
    # Speech-to-Text functionality
    def speech_to_text(audio_bytes):
        # Save audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Use SpeechRecognition to recognize speech
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
                try:
                    # Recognize speech in the selected language
                    original_text = recognizer.recognize_google(audio, language=selected_language_code)
                    return original_text
                except sr.UnknownValueError:
                    st.info("Sorry, could not understand audio.")
                    return ""
                except sr.RequestError as e:
                    st.info(f"Could not request results; {e}")
                    return ""
        finally:
            # Clean up the temporary file
            os.unlink(temp_audio_path)


    col1, col2 = st.columns([12, 1])

    with col1:
        question = st.chat_input(
            "Ask me anything about the machines...", 
            disabled=st.session_state.processing,
            key="chat_input"
        )
    
    with col2:
        audio_bytes=None
        if st.session_state.processing:
        # Show stop button when processing
            if st.button("🟥", key="stop_button", on_click=stop_processing):
                pass
        else:
            # Show microphone when not processing
            audio_bytes = audio_recorder(
                text="",
                recording_color="#ff0000",
                neutral_color="#0083ff",
                icon_name="microphone",
                icon_size="2xl",
                pause_threshold=1,
                key="audio_recorder"
            )

    # Handle microphone input
    if audio_bytes and audio_bytes != st.session_state.audio_bytes:
        st.session_state.audio_bytes = audio_bytes
        
        # Capture original speech
        original_speech = speech_to_text(audio_bytes)
        
        if original_speech:          
            # Set the translated speech as the question
            question = original_speech
    
    # Process the question
    if question and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.rerun()
    
    # If we're processing and have a question in history
    if st.session_state.processing and st.session_state.chat_history:
        try:
            last_question = next(msg["content"] for msg in reversed(st.session_state.chat_history) 
                               if msg["role"] == "user")
            
            sql_response = get_gemini_response(
                last_question, 
                st.session_state.chat_history,
                st.session_state.context
            )
            
            cleaned_sql = clean_sql_query(sql_response)
            translated_speech = translate_to_english(last_question, selected_language_code)
            print("Generated Query: ", cleaned_sql)
            data, columns,sql = read_sql_query(cleaned_sql, os.getenv("MYSQL_DB"),last_question,translated_speech,retry_count=0, max_retries=2)
            
            machine_id_match = re.search(r'(?i)(?:IM\d+)', last_question)
            machine_id = machine_id_match.group(0) if machine_id_match else None
            
            new_context = extract_query_context(sql)
            new_context['query_type'] = {
                'type': 'machine_query',
                'machine_id': machine_id
            } if machine_id else 'general'
            
            st.session_state.context.update(new_context)
            
            formatted_response = format_response(data, columns, new_context['query_type'],st.session_state.chat_history, last_question,sql,selected_language,translated_speech)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": formatted_response
            })
            
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}"
            })
        
        finally:
            st.session_state.processing = False
            st.rerun()
            
if __name__ == "__main__":
    main()
    init_app()