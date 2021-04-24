# These are methods for loading raw eeg data.
# From sets:
#  Motor Imagery - "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\
#  SSVEP -
#  P300 -

from mne.io import concatenate_raws, read_raw_edf


# This just sets our values for what kind of data we want.
# 1 = left/right hands real.
# 2 = left/right hands imagery.
# 3 = hands/feet real.
# 4 = hands/feet imagery.
def __get_runs(test_type):
    run = None
    if test_type == 1:
        run = 3
    elif test_type == 2:
        run = 4
    elif test_type == 3:
        run = 5
    elif test_type == 4:
        run = 6
    else:
        return
    return run


# Gets the all of the runs for a particular set for a given subject.
# Expects subjects in the form "001", "014" etc.
def get_single_mi(subject, test_type):
    # First, we check which set is chosen.
    run = __get_runs(test_type)

    # Now, we grab the data to return
    file = ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
            + subject + "\\S" + subject + "R" + "{:02d}".format(run) + ".edf",
            "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
            + subject + "\\S" + subject + "R" + "{:02d}".format(run + 4) + ".edf",
            "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
            + subject + "\\S" + subject + "R" + "{:02d}".format(run + 8) + ".edf"]
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in file])
    return raw


# Gets all of the runs for a set of subjects.
# expects a list in the form ["001", "002", "003", ...]
def get_multiple_mi(subjects, test_type):
    # first, check the set
    run = __get_runs(test_type)
    files = []

    # Then grab for each subject.
    for subject in subjects:
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
                  + subject + "\\S" + subject + "R" + "{:02d}".format(run) + ".edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
                  + subject + "\\S" + subject + "R" + "{:02d}".format(run + 4) + ".edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
                  + subject + "\\S" + subject + "R" + "{:02d}".format(run + 8) + ".edf"]
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    return raw


# Gets all of the runs for subjects between two values.
# Accepts a list of exclusions for situations with bad data.
# Exclusions should be in form ["001", "002", "003", ...]
def get_all_mi_between(start, end, test_type, exclusions):
    # Check the set
    run = __get_runs(test_type)
    files = []

    # And grab each subject between our start and end values.
    for val in range(start, end):
        # If it's an excluded value, let's skip.
        subject = "{:03d}".format(val)
        if subject in exclusions:
            continue
        # Otherwise, add our files.
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
                  + subject + "\\S" + subject + "R" + "{:02d}".format(run) + ".edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
                  + subject + "\\S" + subject + "R" + "{:02d}".format(run + 4) + ".edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S"
                  + subject + "\\S" + subject + "R" + "{:02d}".format(run + 8) + ".edf"]
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    return raw
