# Real-Time Monitoring of Door Status in Public Transit Systems
## Overview
### Introduction
This is the final project competition of the Computer Vision Course (Spring 2024, National Taiwan University) sponsored by Vivotek. 

### Registration of the Competition
Only the team leader can register. 

### Motivation
An Automated Passenger Counter (APC) is an electronic device installed on transit vehicles, such as buses and trains, to record the times and locations of passenger boarding and disembarking. This data is crucial for analyzing travel patterns and enhancing the operational efficiency of transportation services.
The APC relies on real-time signals from vehicle doors to determine when they open and close, initiating and finalizing the counting process accordingly. However, integrating APC systems with the door status signals of older public transit vehicles can be challenging due to difficulties in accessing and correctly connecting the necessary wiring.
To overcome this, there is a need for a vision-based automatic door status monitoring technology that eliminates the dependency on external wiring for door status signals.

### Challenge of door status monitoring
Generalized door detection and localization using camera video
Accurate monitoring of door statuses, which include:
`Closed/Opening/Open/Closing`

The system must also be robust against various interferences
1. Occlusions caused by passengers.
2. Variations in lighting conditions, including both strong daylight and low nighttime light.
3. Movement of the vehicle.
4. Reflections from glass surfaces.

### Implementation
The executable program should have the capability to:
1. Scan video files in a specified folder (*.mp4).
2. Analyze the video files with door opening/closing events.
3. Generate a JSON file with the guessed frame number of the input video for door opening/closing events.


### Notes
Imporatant Dates
- Evaluation Server Open
    2024/05/10 00:00 GMT+8
- Evaluation Server Close
    2024/06/07 23:59 GMT+8

## Evaluation
We will evaluate Accuracty on 10 testing videos.

### Metric
The APC (Automated Passenger Counter) relies on real-time signals from vehicle doors to determine when they open and close, initiating and finalizing the counting process accordingly.
We use F_score and Inference speed as our main evaluation metric.
[What is F_score? ](https://zh.wikipedia.org/zh-tw/F-score)

The total amount in ground truth => true positive (TP) + false negative (FN)
The total amount you guessed => true positive (TP) + false positive (FP)
The intersection of the above two => true positive (TP)

Ranking order based on
1. F_score = Harmonic average of precision and recall
2. Recall = TP / (TP + FP)
3. Precision = TP / (TP + FN)
4. Inference speed (After Submission on NTU COOL)

## Terms and Conditions
### Submission Policy
Submissions must be made before the end of phase 1. You may submit 5 submissions every day and 150 in total.

Only three sample video files are provided for generating your algorithms. The test video files should be used only for reporting the final results compared to other approaches.
The vision-based automatic door status monitoring technology needs to self-configure without manual input to identify door locations.

Naming your JSON file as output.json
Put output.json into a file named solution.
Upload solution.zip to codalalab to display the ranking.

solution/
├── output.json

For the format of output.json, please refer to Samples/algorithm_output.json