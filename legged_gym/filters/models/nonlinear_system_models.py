from typing import Tuple, cast

import torch
import torch.nn as nn

import torchfilter
from torchfilter import types

torch.random.manual_seed(0)


class NonlinearDynamicsModel(torchfilter.base.DynamicsModel):
    """Forward model for our nonlinear system. Maps (initial_states, controls) pairs to
    (predicted_state, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim, 
                    control_dim, 
                    observation_dim,
                    q_val, dt, device,
                    trainable: bool = False):
        super().__init__(state_dim=state_dim)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.dt = dt 
        self.device = device

        self.Q_tril = torch.eye(state_dim, device=self.device, requires_grad=False) * q_val

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1], device=self.device))

    def forward(
        self,
        *,
        initial_states: types.StatesTorch,
        controls: types.ControlsTorch,
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
        """Forward step for a discrete linear dynamical system.

        Args:
            initial_states (torch.Tensor): Initial states of our system.
            controls (dict or torch.Tensor): Control inputs. Should be either a
                dict of tensors or tensor of size `(N, ...)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted states & uncertainties.
                - States should have shape `(N, state_dim).`
                - Uncertainties should be lower triangular, and should have shape
                `(N, state_dim, state_dim).`
        """

        # Controls should be tensor, not dictionary
        assert isinstance(controls, torch.Tensor)
        controls = cast(torch.Tensor, controls)
        # Check shapes
        N, state_dim = initial_states.shape
        N_alt, control_dim = controls.shape
        assert N == N_alt

        # Compute/return states and noise values
        xrel = (initial_states[:, 0] - self.dt * controls[:, 0] @ torch.cos(initial_states[:, -1])).unsqueeze(-1)
        yrel = (initial_states[:, 1] - self.dt * controls[:, 0] @ torch.sin(initial_states[:, -1])).unsqueeze(-1)
        zrel = initial_states[:, 2].unsqueeze(-1)
        dtheta = (initial_states[:, 3] + self.dt * controls[:, 1]).unsqueeze(-1)
        predicted_states = torch.cat((xrel,
                                      yrel,
                                      zrel,
                                      dtheta
                                      ), dim=-1)
        # predicted_states = (self.A[None, :, :] @ initial_states[:, :, None]).squeeze(-1) + (
        #     self.B[None, :, :] @ controls[:, :, None]
        # ).squeeze(-1)

        # Add output bias if trainable
        if self.trainable:
            predicted_states += self.output_bias

        return predicted_states, self.Q_tril[None, :, :].expand((N, state_dim, state_dim))


class NonlinearKalmanFilterMeasurementModel(torchfilter.base.KalmanFilterMeasurementModel):
    """Kalman filter measurement model for our nonlinear system. Maps states to
    (observation, uncertainty) pairs.

    Args:
        trainable (bool, optional): Set `True` to add a trainable bias to our outputs.
    """

    def __init__(self, state_dim, 
                    observation_dim, 
                    C, r_val,
                    device,
                    trainable: bool = False):
        super().__init__(state_dim=state_dim, observation_dim=observation_dim)
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.device = device

        self.C = C 
        self.C_pinv = torch.pinverse(C)
        self.R_tril = torch.eye(observation_dim, device=self.device, requires_grad=False) * r_val

        # For training tests: if we want to learn this model, add an output bias
        # parameter so there's something to compute gradients for
        self.trainable = trainable
        if trainable:
            self.output_bias = nn.Parameter(torch.FloatTensor([0.1], device=self.device))

    def forward(
        self, *, states: types.StatesTorch
    ) -> Tuple[types.ObservationsNoDictTorch, types.ScaleTrilTorch]:
        """Observation model forward pass, over batch size `N`.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, state_dim)`.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing expected observations
            and cholesky decomposition of covariance.  Shape should be `(N, M)`.
        """
        # Check shape
        N = states.shape[0]
        assert states.shape == (N, self.state_dim)

        # Compute output
        observations = (self.C[None, :, :] @ states[:, :, None]).squeeze(-1)
        scale_tril = self.R_tril[None, :, :].expand((N, self.observation_dim, self.observation_dim))

        # Add output bias if trainable
        if self.trainable:
            observations += self.output_bias

        # Compute/return predicted measurement and noise values
        return observations, scale_tril


def get_trainable_model_error(model) -> float:
    """Get the error of our toy trainable models, which all output the correct output +
    a trainable output bias.

    Returns:
        float: Error. Computed as the absolute value of the output bias.
    """

    # Check model validity -- see above for implementation details
    assert hasattr(model, "trainable")
    assert model.trainable is True
    assert hasattr(model, "output_bias")
    assert model.output_bias.shape == (1,)

    # The error is just absolute value of our scalar output bias
    return abs(float(model.output_bias[0]))
