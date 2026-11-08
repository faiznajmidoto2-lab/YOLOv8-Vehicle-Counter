![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

# YOLOv8 Two-Lane Vehicle Counter

##  Project Background
Traffic monitoring is a critical component of urban transport management. For agencies like Prasarana, understanding vehicle flow on roads and around depots directly informs route planning, scheduling, and infrastructure decisions.

This project is a **real-time vehicle counting system** that uses computer vision and deep learning to detect, track, and count vehicles crossing two defined lanes in a video feed. It provides a practical, offline tool for extracting traffic data without relying on expensive sensor infrastructure.

During my internship at the Malaysian Institute of Road Safety Research (MIROS), I observed that staff were relying on manual, classic methods (tally counters and manual video review) to count vehicles for road safety studies. This process was time-consuming, prone to human error, and inefficient.

I developed this real-time vehicle counting system to automate the process. By leveraging YOLOv8 and ByteTrack, the tool allows researchers to load a traffic video, define lanes, and instantly get automated counts exported to CSV—saving hours of manual labor and providing more accurate data for road safety analysis.

##  Key Features
- **Object Detection:** Uses YOLOv8 (via Ultralytics) to detect cars, motorcycles, buses, and trucks.
- **Multi-Object Tracking:** Integrated ByteTrack algorithm to maintain vehicle identities across frames.
- **Occlusion Handling:** Predicts positions of vehicles temporarily hidden from view using velocity estimation.
- **Two-Lane Counting:** Configurable lines for counting UP and DOWN traffic separately.
- **Motorcycle Boost Mode:** A secondary low-confidence detection pass specifically to catch motorcycles, which are often missed by standard detectors.
- **Modern GUI:** Built with Tkinter and ttkbootstrap, featuring live progress bars, lane configuration sliders, and real-time counts.
- **Data Export:** Exports all counts to CSV for further analysis in tools like Excel or Power BI.

##  Screenshot
![GUI](detection/gui_preview.png)

##  How It Works
![System Architecture](detection/project_architecture.png)
*The system combines YOLOv8 for object detection, ByteTrack for trajectory tracking, and geometric line-crossing logic to accurately count vehicles across two lanes.*

## Tech Stack
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **Tracking:** ByteTrack
- **GUI:** Tkinter, ttkbootstrap
- **Data Handling:** Pandas, NumPy
