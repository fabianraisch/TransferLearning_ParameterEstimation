import torch
import torch.nn as nn

class Physical(nn.Module):
    '''
    Parent class for all Physical models. A Physical is an implementation of a differential equation, in our case different
    RC circuit architectures. This Parent class wraps the mathematical implementation of a physical system in a torch module
    for torch to be able to differentiate and adds some utility function.

    '''
    tame = 10e4
    last_hidden_init = None
    def __init__(self):
        super().__init__()

    def reset(self):
        # resets any hidden states of more complex RC architectures (like 2R2C)
        pass

    def tame_values(self, x):
        # utility function that limits the effective derivatives of all dynamic values to about tame/10 (per dt)
        return (x*self.tame)/torch.sqrt(x**2 + self.tame**2)

    def set_params(self, params):
        # sets the physicals parameters to integrate after
        self.params = params

    def forward(self, t, state):
        # forward function necessary for the Physical to be a torch object
        out = self.tame_values(self.step(state, self.external_data[int(t.item())], self.params, int(self.dt)))
        return out

class RC(Physical):
    '''
    Implementation of the simplest RC architecture, one capacitance for the lumped inner temperature
    and one resistor for the lumped mass of the walls, roof, floor etc.

    '''
    parameter_names = ["C", "R", "A_eff"]   # defines the amount and locations of the parameters inside the density args of step()
    dynamic_variables = 1                   # defines which subset of data (data[:dynamic_variables]) is to be simulated rather than external
    data_names = ["T_in", "T_out", "heatPower", "solarGains"] # format of the loaded data, used for plotting

    def __init__(self):
        super().__init__()

    def step(self, densities, data, params, dt: torch.Tensor):
        # this function calculates one dT timestep. The integration is managed by odeint in the Physical class
        # densities: [T_in]
        # data: [T_ambient, heatPower, Q_solar]
        # params: [C, R, A_eff]

        T_in = densities[0]
        T_ambient = data[0]
        heatPower = data[1]
        Q_solar = data[2]

        C = params[0]
        R = params[1]
        A_eff = params[2]

        dT = dt/C * ((T_ambient - T_in)/R + heatPower + Q_solar*A_eff)

        return dT.unsqueeze(0)

    def initial_condition(self, cfg, wdata):
        # read the initial condition for initial data generation. Not used in the optimization loop.
        return [cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]



class TwoR2C(Physical):
    '''
    Implementation of a more advanced RC architecture:
    - One capacitance for the lumped inner temperature
    - One resistor for the heat transfer between the walls and the inside
    - Another capacitance for the wall's mass and temperature
    - Another resistor for the heat transfer between the outside and the walls.

    Note that the wall's temperature is not given by data and thus is estimated at the beginning of each simulation.
    '''
    parameter_names = ["C1", "R1", "C2", "R2", "A_eff"] # defines the amount and locations of the parameters inside the density args of step()
    dynamic_variables = 1                               # defines which subset of data (data[:dynamic_variables]) is to be simulated rather than external
    data_names = ["T_in", "T_out", "heatPower", "solarGains"] # format of the loaded data, used for plotting

    def __init__(self):
        super().__init__()
        self.T_envelope = None

    def reset(self):
        # reset the hidden state of the wall to None for self.step() to recognize and re-initialize.
        self.T_envelope = None

    def step(self, densities, data, params, dt):
        # this function calculates one dT timestep. The integration is managed by odeint in the Physical class
        # densities: [T_in]
        # data: [T_ambient, heatPower, Q_solar]
        # params: [C, R, A_eff]

        T_in = densities[0]
        T_ambient = data[0]
        heatPower = data[1]
        Q_solar = data[2]

        C1 = params[0] # Inside capacity of building
        R1 = params[1] # Inner Resistance of envelope
        C2 = params[2] # Capacity of envelope
        R2 = params[3] # Outer Resistance of envelope
        A_eff = params[4] # Effectve window size

        # if the hidden stated has to be initialized, estimate wall Temperature to be weighted mean of T_in and T_ambient
        if self.T_envelope is None:
            self.T_envelope = T_ambient + (T_in - T_ambient)* R2 / (R1 + R2) # "voltage divider" 
            self.last_hidden_init = self.T_envelope.detach().item()         # log hidden state initialization to possibly plot later

        P_in2Wall = (T_in - self.T_envelope) / R1
        dT_env = (P_in2Wall + (T_ambient - self.T_envelope) / R2) * dt / C2 # (power from inside into env + power from env to outside)*dt/C2
        self.T_envelope = self.T_envelope + self.tame_values(dT_env)        # manually integrate hidden state since it is not passed to odeint
        dT_in = (heatPower + Q_solar * A_eff - P_in2Wall) * dt / C1         # (power that enters the inside - power that is lost into the wall)*dt/C1  

        return (dT_in).unsqueeze(0)


    def initial_condition(self, cfg, wdata):
        # read the initial condition for initial data generation. Not used in the optimization loop.
        return [cfg["initial_conditions"]["T_in"], wdata[0, 2] + 273.15, 0, 0]
