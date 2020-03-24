# Research Repository for Hand-Activities
This is the research repository for the CHI 2019 Paper: "Sensing Fine-Grained Hand Activity with Smartwatches." This repository contains links and instructions to a subset of our dataset.

# System Requirements
The system is written in `python3`, specifically `tensorflow` and `keras`.

To begin, we recommend using `virtualenv` to run a self-contained setup:
```bash
$ virtualenv ./hand-activities -p python3.6
$ source hand-activities/bin/activate
```

Once `virtualenv` is activated, install the following dependencies via `pip`:

```bash
(hand-activities)$ git clone https://github.com/FIGLAB/hand-activities.git
(hand-activities)$ pip install numpy
(hand-activities)$ pip install tensorflow
(hand-activities)$ pip install keras
(hand-activities)$ pip install wget
```

# Dataset
Download the dataset here:
https://www.dropbox.com/sh/vawocn8ae2eiy96/AAAA7rgUCSp4RIXLixARh1Y8a?dl=1

Unzip each file, and you’ll get numpy arrays that are organized by user or by round.

```
data = np.load('path_to_features_file.npy’)
print(data.shape)
```

You’ll have matched label array pairs per condition. For example:

```
round1_x = np.load(‘round1_features_X.npy’)
round1_y = np.load(‘round1_features_Y.npy’)
round1_labels = np.load(‘round1_features_labels.npy’)

print(round1_x.shape)
print(round1_y.shape)
print(round1_labels.shape)
```

If you print dataset shapes, their row counts should line up. X is a `3 x 256 x 48` array (x, y, z axes of the high-speed accelerometer signal).

Each row is a datapoint. You can assemble your train / test / validation splits by concatenating different conditions.

We suggest extracting a row from the input features (X), and plotting then using python `matplotlib`.

## Disclaimer

```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```
