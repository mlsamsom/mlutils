import numpy as np


class TimeSeriesData():
    """Generates a sine curve"""
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = float(xmax - xmin)/float( num_points )
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        # grab random starting point for each batch
        rand_start = np.random.rand(batch_size, 1)

        # convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))

        # Create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution

        # create y data
        y_batch = np.sin(batch_ts)

        # format for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)
