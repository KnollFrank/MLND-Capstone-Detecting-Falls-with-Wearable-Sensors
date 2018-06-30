# Machine Learning Engineer Nanodegree

## Detecting Falls with Wearable Sensors

## Capstone Proposal
Frank Knoll  
30th June, 2018

## Proposal

### Domain Background

One day you will be 65 years old. Then according to the World Health Organization ([1]) your chance of falling once a year is 28–35% with serious consequences such as heavy injuries. Imagine carrying a sensor or smartphone near your waist or thigh which recognizes your fall and immediately alerts a person to help you, that would be great. But imagine further the sensor would confuse one of your Activities of Daily Living (ADLs) such as sitting, standing or walking with a fall, then the person to help you would have been alarmed without reason. Therefore such a sensor based fall detection system should not miss any falls and should also trigger no false alarms when performing ADLs.

### Problem Statement

The problem to be solved is to distinguish falls from activities of daily living using a wireless sensor unit fitted to a person's waist.
There are numerous types of falls and activities of daily living as can be seen in the following tables, but the task is just to binary classify actions in falls and non-falls. A challenge in this classification task is not to confuse some of the non-fall actions, which are high-impact events, with falls.

#### Fall Actions:

\#  | Label                  | Description
---|------------------------|---------------------------------------------------------------------------
1  | front-lying            | from vertical falling forward to the floor
2  | front-protecting-lying | from vertical falling forward to the floor with arm protection
3  | front-knees            | from vertical falling down on the knees
4  | front-knees-lying      | from vertical falling down on the knees and then lying on the floor
5  | front-right            | from vertical falling down on the floor, ending in right lateral position
6  | front-left             | from vertical falling down on the floor, ending in left lateral position
7  | front-quick-recovery   | from vertical falling on the floor and quick recovery
8  | front-slow-recovery    | from vertical falling on the floor and slow recovery
9  | back-sitting           | from vertical falling on the floor, ending sitting
10 | back-lying             | from vertical falling on the floor, ending lying
11 | back-right             | from vertical falling on the floor, ending lying in right lateral position
12 | back-left              | from vertical falling on the floor, ending lying in left lateral position
13 | right-sideway          | from vertical falling on the floor, ending lying
14 | right-recovery         | from vertical falling on the floor with subsequent recovery
15 | left-sideway           | from vertical falling on the floor, ending lying
16 | left-recovery          | from vertical falling on the floor with subsequent recovery
17 | syncope                | from standing falling on the floor following a vertical trajectory
18 | syncope-wall           | from standing falling down slowly slipping on a wall
19 | podium                 | from vertical standing on a podium going on the floor
20 | rolling-out-bed        | from lying, rolling out of bed and going on the floor

#### Non-Fall Actions (ADLs):

\#  | Label             | Description
---|-------------------|---------------------------------------------------------------------------------
21 | lying-bed         | from vertical lying on the bed
22 | rising-bed        | from lying to sitting
23 | sit-bed           | from vertical to sitting with a certain acceleration onto a bed (soft surface)
24 | sit-chair         | from vertical to sitting with a certain acceleration onto a chair (hard surface)
25 | sit-sofa          | from vertical to sitting with a certain acceleration onto a sofa (soft surface)
26 | sit-air           | from vertical to sitting in the air exploiting the muscles of legs
27 | walking-fw        | walking forward
28 | jogging           | running
29 | walking-bw        | walking backward
30 | bending           | bending about 90 degrees
31 | bending-pick-up   | bending to pick up an object on the floor
32 | stumble           | stumbling with recovery
33 | limp              | walking with a limp
34 | squatting-down    | squatting, then standing up
35 | trip-over         | bending while walking and then continuing walking
36 | coughing-sneezing | coughing or sneezing

### Datasets and Inputs

Seven males and seven females participated in a study. A wireless sensor unit was fitted to the subject's waist among other body parts. The sensor unit comprises three tri-axial devices: accelerometer, gyroscope, and  magnetometer/compass. Raw motion data was recorded along three perpendicular axes (x, y, z) from the unit with a sampling frequency of 25 Hz. A set of trials consists of 20 fall actions (see table 'Fall Actions' above) and 16 ADLs (see table 'Non-Fall Actions' above). Each trial lasted about 15s on average. The 14 volunteers repeated each test for five times. Then the peak of the total acceleration vector was detected, and two seconds of the sequence before and after the peak acceleration were kept. As an example, the first five records of the file `FallDataSet/101/Testler Export/901/Test_1/340535.txt` which contains the recorded data from the waist sensor attached to a male while he was falling from vertical forward to the floor, look like this:

| Acc_X ($m/s^2$) | Acc_Y ($m/s^2$) | Acc_Z ($m/s^2$) | Gyr_X (°/s) | Gyr_Y (°/s) | Gyr_Z (°/s) | Mag_X (Gauss) | Mag_Y (Gauss) | Mag_Z (Gauss) |
|-----------------|-----------------|-----------------|-------------|-------------|-------------|---------------|---------------|---------------|
| 9.715           | 1.121           | 0.947           | 0.004       | -0.004      | -0.004      | -0.818        | 0.515         | 0.012         |
| 9.733           | 1.131           | 0.985           | -0.014      | -0.006      | 0.000       | -0.818        | 0.517         | 0.009         |
| 9.750           | 1.103           | 0.941           | -0.003      | 0.004       | 0.000       | -0.821        | 0.517         | 0.009         |
| 9.745           | 1.111           | 0.999           | 0.001       | 0.005       | -0.004      | -0.821        | 0.515         | 0.009         |
| 9.725           | 1.100           | 1.019           | -0.005      | 0.006       | -0.008      | -0.821        | 0.517         | 0.000         |
| $\vdots$        | $\vdots$        | $\vdots$        | $\vdots$    | $\vdots$    | $\vdots$    | $\vdots$      | $\vdots$      | $\vdots$      |

The FallDataSet kann be downloaded from

https://drive.google.com/open?id=1gqS1fkTvtuAaKj_0cn9n04ng1qDAoZ2t.

### Solution Statement

The intendet solution is to train a machine learning classifier on the FallDataSet in order to distinguish falls from activities of daily living.

### Benchmark Model

As a benchmark model the following classsifiers are used:

- random classifier
- classifier which classifies every action as a fall
- classifier which classifies every action as an activity of daily living

### Evaluation Metrics

In distinguishing falls from ADLs, the following conditions must be met:

- "False negatives, which indicate missed falls, must be avoided by all means, since user manipulation may not be possible if a fall results in physical and/or mental impairment." ([2])
- "False alarms (false positives) caused by misclassified ADLs, although a nuisance, can be canceled by the user." ([2])

So an evaluation metric should be chosen which punishes false negatives more than false positives. As can be seen from the formula of the $F_\beta$-score (https://en.wikipedia.org/wiki/F1_score) $F_\beta = \frac {(1 + \beta^2) \cdot \mathrm{true\ positive} }{(1 + \beta^2) \cdot \mathrm{true\ positive} + \beta^2 \cdot \mathrm{false\ negative} + \mathrm{false\ positive}}\,$ this can be achieved by setting $\beta>1$, e.g. $\beta = 2$. So the chosen evaluation metric is the $F_2$-score.

### Project Design

Following [2] the first task is to perform feature extraction. From the raw data Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Mag_X, Mag_Y and Mag_Z of the FallDataSet the following features are extracted: minimum, maximum, mean, skewness, kurtosis, the first 11 values of the autocorrelation sequence and the first five frequencies with maximum magnitude of the discrete Fourier transform (DFT) along with the five corresponding amplitudes, resulting in a feature vector of dimensionality 234 (26 features for each one of the nine measured signals) for each test.

To be more specific, let $s = [s_1, s_2,\dots, s_N]^T$ be the raw data of a signal (e.g. the column Acc_X in the table above). Then the extracted features for this signal are defined as follows:

- $\operatorname{mean}(s) = \mu = \frac{1}{N} \sum_{n=1}^{N} s_n$
- $\operatorname{variance}(s) = \sigma^2 = \frac{1}{N} \sum_{n=1}^{N} (s_n-\mu)^2$
- $\operatorname{skewness}(s) = \frac{1}{N \sigma^3} \sum_{n=1}^{N} (s_n-\mu)^3$
- $\operatorname{kurtosis}(s) = \frac{1}{N \sigma^4} \sum_{n=1}^{N} (s_n-\mu)^4$
- $\operatorname{autocorrelation}(s) = \frac{1}{N - \Delta} \sum_{n=0}^{N - \Delta - 1} (s_n-\mu)(s_{n - \Delta} - \mu)$. where $\Delta = 0, 1, \dots, N-1$
- $\operatorname{DFT}_q(s) = \sum_{n=0}^{N-1} s_n e^{- \frac{j 2 \pi q n}{N}}$, where $q = 0, 1, \dots, N-1$

$\operatorname{DFT}_q(s)$ is the $q$th element of the 1-D $N$-point DFT.

Then the following classsifiers are applied to the feature data set using 10-fold cross validation: Decision Tree, K-Nearest Neighbors, Random Forest and Support Vector Machine. The classifier having the best $F_2$-score will be taken as the solution to the problem.

------

- [1] World Health Organization: Global report on falls prevention in older age.
http://www.who.int/ageing/publications/Falls_prevention7March.pdf
- [2] Özdemir, Ahmet Turan, and Billur Barshan. “Detecting Falls with Wearable Sensors Using Machine Learning Techniques.” Sensors (Basel, Switzerland) 14.6 (2014): 10691–10708. PMC. Web. 23 Apr. 2017. http://www.mdpi.com/1424-8220/14/6/10691/pdf
- [3] Ntanasis P., Pippa E., Özdemir A.T., Barshan B., Megalooikonomou V., "Investigation of sensor placement for accurate fall detection", 6th EAI International Conference on Wireless Mobile Communication and Healthcare (MobiHealth), Milan, Italy, 14-16 Nov. 2016, pp.1-6. https://www.researchgate.net/profile/Billur_Barshan/publication/318146579_Investigation_of_Sensor_Placement_for_Accurate_Fall_Detection/links/5999a8a745851564432dbdf7/Investigation-of-Sensor-Placement-for-Accurate-Fall-Detection.pdf?origin=publication_detail
