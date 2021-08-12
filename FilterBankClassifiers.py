#This is an attempt to implement the filter-bank approach to processing and classifying eeg data.
#Sources for examples:
#   https://neurotechx.github.io/eeg-notebooks/auto_examples/visual_ssvep/02r__ssvep_decoding.html
#   https://github.com/NeuroTechX/moabb/blob/be1f81220869158ef37e1ab91b0279fe60aeed5b/moabb/paradigms/base.py#L51

#General idea is to grab the raw data, and for each filter band we copy it, filter it, epoch it
#and append it to our data. Then, we rearrange the data using np.array(X).transpose((1, 2, 3, 0))
#to take it from form (filter, epoch, channel, time) to (epoch, channel, time, filter).
#Then, we can use the FBCSP method from moabb. Otherwise, we can use it elsewhere, such as a CNN maybe?

def split_to_bands(raw, filter_bands=[[[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]],
                   tmin=0, tmax=4, ch_list=[]):
    return

