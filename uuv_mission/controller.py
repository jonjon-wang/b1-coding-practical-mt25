class PDController:
    """Discrete-time PD controller.

    The controller implements u[t] = kp * e[t] + kd * (e[t] - e[t-1]).
    returns the control action and stores the last error for the next call.
    """
    def __init__(self, kp = 0.15, kd  = 0.6, last_error = 0.0):
        '''Simple PD controller implementation, initializes with given gains and last error'''
        self.kp = kp
        self.kd = kd
        self.last_error = last_error

    def reset(self):
        '''Reset the last error to zero before a new simulation.'''
        self.last_error = 0.0

    def update(self, error: float) -> float:
        '''
        Compute the control action based on the current error.
        u[t] = kp * e[t] + kd * (e[t] - e[t-1])
        '''
        derivative = error - self.last_error
        u = self.kp * error + self.kd * derivative
        self.last_error = error
        return u