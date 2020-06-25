import numpy as np
from .timers import Timer

class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, dvs, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        ts = dvs['ts'][:, np.newaxis]
        x = dvs['x'][:, np.newaxis]
        y = dvs['y'][:, np.newaxis]
        pol = dvs['pol'][:, np.newaxis]
        self.data = np.concatenate((ts, x, y, pol), axis=1)
        self.start_index = start_index
        self.num_events = num_events
        
    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            end_index = self.start_index + self.num_events
            if end_index >= self.data.shape[0]:
                raise StopIteration
            event_window = self.data[self.start_index:end_index, :]
            self.start_index = end_index
            return event_window
    
class FixedDurationEventReader:

    def __init__(self, dvs, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        self.ts = dvs['ts'][:, np.newaxis]
        x = dvs['x'][:, np.newaxis]
        y = dvs['y'][:, np.newaxis]
        pol = dvs['pol'][:, np.newaxis]
        self.data = np.concatenate((self.ts, x, y, pol), axis=1)
        self.start_index = start_index
        self.last_stamp = self.ts[start_index]
        self.duration_s = duration_ms / 1000.0

    def __iter__(self):
        return self

    def __del__(self):
        pass

    def __next__(self):
        with Timer('Reading event window from container'):
            self.last_stamp += self.duration_s
            end_index = self.ts.searchsorted(self.last_stamp)
            if end_index == self.ts.shape[0]:
                raise StopIteration
            event_window = self.data[self.start_index:end_index, :]
            self.start_index = end_index
            return event_window

    